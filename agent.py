"""
agent.py
========
ReAct-style groundwater investigation system built with LangGraph.

The LLM drives tool selection at runtime.
Python does not decide which tool to call or when — the LLM does.

Graph:
    START
      ↓
    investigator (LLM + tools bound)
      ├── tool_calls present → tools → investigator  (ReAct loop)
      └── no tool_calls     → communicator → END

The investigator LLM:
    - Sees the full conversation history including previous tool results
    - Decides which tool to call next (or stops)
    - Decides whether borderline predictions need sensitivity checks
    - Decides which features need context lookups
    This is what makes it agentic — not hardcoded if/else.

LangSmith: set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY
Every graph invocation is traced automatically.
"""

import json
import os
import operator
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)

from reasoning import FEATURE_INFO, FEATURE_IMPORTANCE, FEATURE_STATS, _level
from llm_reasoning import generate_llm_reasoning
from train import normalise_shap

# ── Artifacts ─────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
MODEL_DIR    = BASE / "model"
_model       = joblib.load(MODEL_DIR / "rf_model.joblib")
_explainer   = joblib.load(MODEL_DIR / "shap_explainer.joblib")
FEATURE_COLS = json.loads((MODEL_DIR / "feature_columns.json").read_text())


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# The LLM invokes these by name. They don't call each other.
# ══════════════════════════════════════════════════════════════════════════════

@tool
def predict_groundwater(feature_values: dict) -> dict:
    """
    Run the trained RandomForest model on the site's feature values.
    Returns prediction (0=absent, 1=present) and probability (0.0-1.0).
    Always call this first before any other tool.
    """
    X = pd.DataFrame([feature_values])[FEATURE_COLS]
    p = float(_model.predict_proba(X)[0, 1])
    return {"prediction": int(p >= 0.5), "probability": round(p, 4)}


@tool
def compute_shap_values(feature_values: dict) -> dict:
    """
    Compute per-prediction SHAP values for the site.
    Positive SHAP = feature pushed toward groundwater present.
    Negative SHAP = feature pushed toward groundwater absent.
    Call this after predict_groundwater to understand what drove the result.
    """
    X   = pd.DataFrame([feature_values])[FEATURE_COLS]
    raw = _explainer.shap_values(X)
    arr = normalise_shap(raw, n_samples=1, n_features=len(FEATURE_COLS))
    return {col: round(float(arr[0, i]), 4) for i, col in enumerate(FEATURE_COLS)}


@tool
def get_feature_context(feature_name: str, value: float) -> dict:
    """
    Check where a feature value sits in the training distribution.
    Returns whether it is low/medium/high and which direction favors groundwater.
    Use this to understand if a SHAP value makes sense given the feature's level.
    """
    if feature_name not in FEATURE_STATS:
        return {"error": f"Unknown feature: {feature_name}"}
    stats    = FEATURE_STATS[feature_name]
    info     = FEATURE_INFO.get(feature_name, {})
    imp_keys = list(FEATURE_IMPORTANCE.keys())
    return {
        "feature":             feature_name,
        "label":               info.get("label", feature_name),
        "value":               value,
        "level":               _level(feature_name, value),
        "mean":                round(stats["mean"], 2),
        "importance_rank":     imp_keys.index(feature_name) + 1 if feature_name in imp_keys else None,
        "favorable_direction": info.get("favorable", "context"),
    }


@tool
def run_sensitivity_check(feature_values: dict, feature_name: str, delta: int) -> dict:
    """
    Test how much the prediction changes if one feature is nudged by delta class steps.
    Use this when probability is borderline (0.38-0.65) or when a feature shows
    a conflict between its level and SHAP direction. delta should be +1 or -1.
    """
    if feature_name not in feature_values:
        return {"error": f"{feature_name} not found in feature_values"}
    stats        = FEATURE_STATS[feature_name]
    original_val = feature_values[feature_name]
    new_val      = max(stats["min"], min(stats["max"], original_val + delta))

    def _p(fv):
        return round(float(_model.predict_proba(pd.DataFrame([fv])[FEATURE_COLS])[0, 1]), 4)

    orig_prob = _p(feature_values)
    new_prob  = _p({**feature_values, feature_name: new_val})
    return {
        "feature": feature_name, "original_val": original_val, "new_val": new_val,
        "original_prob": orig_prob, "new_prob": new_prob,
        "impact": round(new_prob - orig_prob, 4),
    }


_TOOLS         = [predict_groundwater, compute_shap_values, get_feature_context, run_sensitivity_check]
_TOOLS_BY_NAME = {t.name: t for t in _TOOLS}


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# messages accumulates the full ReAct conversation.
# Structured fields accumulate tool results for the communicator.
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:             Annotated[list, operator.add]
    feature_values:       dict
    session_id:           str
    session_summary:      dict | None
    agg_stats:            dict | None
    place_name:           str  | None
    prediction:           int  | None
    probability:          float | None
    shap_values:          dict  | None
    sensitivity_results:  Annotated[list, operator.add]
    flagged_conflicts:    Annotated[list, operator.add]
    investigation_notes:  Annotated[list, operator.add]
    investigation_report: str  | None
    final_reasoning:      str  | None
    reasoning_source:     str  | None
    reasoning_model:      str  | None


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK LLM
# Used when HF_API_TOKEN is not configured.
# Follows the same graph structure — only the decision-maker changes.
# ══════════════════════════════════════════════════════════════════════════════

class _FallbackLLM:
    """
    Deterministic tool caller for when no LLM is configured.
    Produces the same AIMessage/ToolMessage flow as a real LLM.
    The graph, routing, and tool execution are identical either way.
    """

    def bind_tools(self, tools):
        return self

    def invoke(self, messages: list) -> AIMessage:
        tool_msgs       = [m for m in messages if isinstance(m, ToolMessage)]
        called_names    = [m.name for m in tool_msgs]
        fetched_context = {
            json.loads(m.content).get("feature")
            for m in tool_msgs
            if m.name == "get_feature_context"
        }
        fv = self._extract_features(messages)

        # Fixed investigation sequence
        if "predict_groundwater" not in called_names:
            return self._call("predict_groundwater", {"feature_values": fv}, "t_pred")

        if "compute_shap_values" not in called_names:
            return self._call("compute_shap_values", {"feature_values": fv}, "t_shap")

        for col in list(FEATURE_IMPORTANCE.keys())[:3]:
            if col not in fetched_context and col in fv:
                return self._call(
                    "get_feature_context",
                    {"feature_name": col, "value": float(fv.get(col, 0))},
                    f"t_ctx_{col}",
                )

        if "run_sensitivity_check" not in called_names:
            pred_msg = next((m for m in tool_msgs if m.name == "predict_groundwater"), None)
            top_col  = list(FEATURE_IMPORTANCE.keys())[0] if FEATURE_IMPORTANCE else None
            if pred_msg and top_col and top_col in fv:
                try:
                    pred_data = json.loads(pred_msg.content)
                    delta     = 1 if pred_data.get("prediction", 1) == 0 else -1
                    return self._call(
                        "run_sensitivity_check",
                        {"feature_values": fv, "feature_name": top_col, "delta": delta},
                        "t_sens",
                    )
                except Exception:
                    pass

        return AIMessage(content="Investigation complete.")

    def _extract_features(self, messages: list) -> dict:
        for m in messages:
            if isinstance(m, HumanMessage):
                try:
                    idx = m.content.find("{")
                    if idx >= 0:
                        return json.loads(m.content[idx:])
                except Exception:
                    pass
        return {}

    def _call(self, name: str, args: dict, call_id: str) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _get_llm():
    """Return a tool-capable LLM, or the deterministic fallback."""
    token = os.environ.get("HF_API_TOKEN", "").strip()
    if not token:
        return _FallbackLLM()
    try:
        from langchain_community.llms import HuggingFaceEndpoint
        from langchain_community.chat_models.huggingface import ChatHuggingFace
        model    = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        endpoint = HuggingFaceEndpoint(
            repo_id=model, huggingfacehub_api_token=token,
            task="text-generation", max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)
    except Exception:
        return _FallbackLLM()


def _system_prompt(state: AgentState) -> str:
    fv      = state["feature_values"]
    session = state.get("session_summary") or {}
    agg     = state.get("agg_stats") or {}

    session_ctx = (
        f"\nSession context: user has checked {session['total_checked']} sites. "
        f"Best so far: {session['highest_probability_site']['place']} "
        f"({session['highest_probability_site']['probability']*100:.1f}%)."
        if session.get("total_checked", 0) > 0 else ""
    )
    agg_ctx = (
        f"\nHistorical: across {agg['total_predictions']} predictions, "
        f"{agg['groundwater_likely_pct']}% were groundwater-likely."
        if agg.get("total_predictions", 0) > 5 else ""
    )

    return f"""You are a hydrogeology investigation agent.
Analyze this site for groundwater well presence.

Feature values:
{json.dumps(fv, indent=2)}
{session_ctx}{agg_ctx}

Available tools:
- predict_groundwater   → always call this first
- compute_shap_values   → call after prediction to see what drove it
- get_feature_context   → call for the 2-3 features with highest absolute SHAP
- run_sensitivity_check → call if probability is 0.38-0.65 (borderline),
                          or if a feature's level conflicts with its SHAP sign

When you have sufficient information, respond with text (no tool call).
Do not repeat a tool call with the same arguments."""


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════

def investigator_node(state: AgentState) -> dict:
    """
    Investigator Agent — the LLM sees the full conversation and decides
    which tool to call next, or signals it is done investigating.
    This is the ReAct reasoning step.
    """
    try:
        llm      = _get_llm()
        response = llm.bind_tools(_TOOLS).invoke(
            [SystemMessage(content=_system_prompt(state))] + state["messages"]
        )
    except Exception as e:
        response = _FallbackLLM().invoke(state["messages"])
        return {
            "messages":          [response],
            "investigation_notes": [f"[Warning] LLM call failed ({e!s:.80}), using fallback"],
        }
    return {"messages": [response]}


def _note(name: str, result: dict) -> str:
    """One-line trace note for the agent trace UI tab."""
    if name == "predict_groundwater":
        return f"prediction={result['prediction']}, probability={result['probability']:.4f}"
    if name == "compute_shap_values":
        top = max(result, key=lambda k: abs(result[k]))
        return f"top driver: {top} (SHAP={result[top]:+.3f})"
    if name == "get_feature_context":
        return f"{result.get('label', result.get('feature'))} is {result.get('level')} (rank #{result.get('importance_rank')})"
    if name == "run_sensitivity_check":
        return (f"{result.get('feature')}: {result.get('original_val')}→{result.get('new_val')}, "
                f"impact {result.get('impact', 0)*100:+.1f}pp")
    return str(result)[:120]


def _find_conflicts(shap: dict, contexts: list[dict]) -> list[str]:
    """Detect features where SHAP direction contradicts the expected favorable direction."""
    out = []
    for ctx in contexts:
        col, fav, lvl = ctx.get("feature"), ctx.get("favorable_direction"), ctx.get("level")
        sv = shap.get(col, 0) if shap else 0
        label = ctx.get("label", col)
        if fav == "high" and lvl == "low"  and sv > 0.05:
            out.append(f"{label} is LOW but SHAP is positive — other features may compensate")
        elif fav == "low" and lvl == "high" and sv < -0.05:
            out.append(f"{label} is HIGH (unfavorable) and SHAP confirms it hurts likelihood")
        elif fav == "high" and lvl == "high" and sv < -0.05:
            out.append(f"{label} is HIGH but SHAP is negative — unusual combination")
    return out


def tool_node(state: AgentState) -> dict:
    """
    Tool executor — runs every tool call in the last AIMessage.
    Updates both the message history (ToolMessage) and structured state
    fields simultaneously, so the communicator has clean data to work with.
    """
    last    = state["messages"][-1]
    updates = {
        "messages":            [],
        "investigation_notes": [],
        "sensitivity_results": [],
        "flagged_conflicts":   [],
    }
    contexts = []

    for call in last.tool_calls:
        name   = call["name"]
        result = _TOOLS_BY_NAME[name].invoke(call["args"])

        updates["messages"].append(ToolMessage(
            content=json.dumps(result), tool_call_id=call["id"], name=name,
        ))
        updates["investigation_notes"].append(f"[Tool: {name}] → {_note(name, result)}")

        if name == "predict_groundwater":
            updates["prediction"]  = result["prediction"]
            updates["probability"] = result["probability"]
        elif name == "compute_shap_values":
            updates["shap_values"] = result
        elif name == "get_feature_context":
            contexts.append(result)
        elif name == "run_sensitivity_check":
            updates["sensitivity_results"].append(result)

    if contexts:
        shap = state.get("shap_values") or updates.get("shap_values")
        for conflict in _find_conflicts(shap, contexts):
            updates["flagged_conflicts"].append(conflict)
            updates["investigation_notes"].append(f"[Conflict] {conflict}")

    return updates


def communicator_node(state: AgentState) -> dict:
    """
    Communicator Agent — builds a structured investigation report from
    everything the Investigator gathered, then calls the LLM reasoning
    engine (Hugging Face or template fallback) to write the final explanation.
    """
    shap  = state.get("shap_values") or {}
    sens  = state.get("sensitivity_results") or []
    conf  = state.get("flagged_conflicts") or []
    pred  = state.get("prediction")
    prob  = state.get("probability")

    shap_str = ", ".join(
        f"{FEATURE_INFO.get(k, {}).get('label', k)}={v:+.3f}"
        for k, v in sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    )
    sens_str = " ".join(
        f"If {FEATURE_INFO.get(s['feature'], {}).get('label', s['feature'])} "
        f"changed {s['original_val']:.0f}→{s['new_val']:.0f}: {s['impact']*100:+.1f}pp."
        for s in sens if abs(s.get("impact", 0)) >= 0.03
    ) or "Prediction is robust to small feature changes."

    report = (
        f"PREDICTION: {'Groundwater likely' if pred == 1 else 'Groundwater unlikely'} "
        f"(probability={prob*100:.1f}%).\n"
        f"TOP SHAP DRIVERS: {shap_str}.\n"
        f"SENSITIVITY: {sens_str}"
        + (f"\nCONFLICTS: {'; '.join(conf)}" if conf else "")
    )

    result = generate_llm_reasoning(
        feature_values=state["feature_values"], prediction=pred, probability=prob,
        place_name=state.get("place_name"), shap_values=shap,
        investigation_report=report, session_summary=state.get("session_summary"),
        agg_stats=state.get("agg_stats"),
    )

    return {
        "investigation_report": report,
        "investigation_notes":  [f"[Communicator] Explanation via {result['source']}"],
        "final_reasoning":      result["text"],
        "reasoning_source":     result["source"],
        "reasoning_model":      result.get("model"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# The single conditional edge that makes this ReAct.
# ══════════════════════════════════════════════════════════════════════════════

def route(state: AgentState) -> Literal["tools", "communicator"]:
    """
    If the LLM produced tool calls → execute them and loop back.
    If the LLM produced plain text → move to the communicator.
    This is where LangGraph drives the agentic loop, not Python if/else.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "communicator"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("investigator", investigator_node)
    g.add_node("tools",        tool_node)
    g.add_node("communicator", communicator_node)
    g.set_entry_point("investigator")
    g.add_conditional_edges("investigator", route, {
        "tools":        "tools",
        "communicator": "communicator",
    })
    g.add_edge("tools",        "investigator")
    g.add_edge("communicator", END)
    return g.compile()


_graph = _build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT — API contract unchanged, main.py needs no edits
# ══════════════════════════════════════════════════════════════════════════════

def run_investigation(
    feature_values:  dict,
    session_id:      str         = "default",
    session_summary: dict | None = None,
    agg_stats:       dict | None = None,
    place_name:      str  | None = None,
) -> dict:
    """
    Run the ReAct investigation graph.
    The only function main.py imports from this module.
    Input/output contract is identical to the previous version.
    """
    final = _graph.invoke({
        "messages": [HumanMessage(content=(
            f"Investigate groundwater presence for these feature values: "
            f"{json.dumps(feature_values)}"
        ))],
        "feature_values":       feature_values,
        "session_id":           session_id,
        "session_summary":      session_summary,
        "agg_stats":            agg_stats,
        "place_name":           place_name,
        "prediction":           None,
        "probability":          None,
        "shap_values":          None,
        "sensitivity_results":  [],
        "flagged_conflicts":    [],
        "investigation_notes":  [],
        "investigation_report": None,
        "final_reasoning":      None,
        "reasoning_source":     None,
        "reasoning_model":      None,
    })

    return {
        "prediction":           final["prediction"],
        "probability":          final["probability"],
        "shap_values":          final["shap_values"],
        "sensitivity_results":  final["sensitivity_results"],
        "flagged_conflicts":    final["flagged_conflicts"],
        "investigation_notes":  final["investigation_notes"],
        "investigation_report": final["investigation_report"],
        "final_reasoning":      final["final_reasoning"],
        "reasoning_source":     final["reasoning_source"],
        "reasoning_model":      final["reasoning_model"],
    }