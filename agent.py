"""
agent.py
========
Multi-agent LangGraph system for groundwater site investigation.

This is the core agentic component of the application. It is called by
main.py (FastAPI) via run_investigation() — that is the only public
entry point. Everything else is internal implementation.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │              Investigator Agent                  │
    │                                                 │
    │  Decides which tools to call based on what      │
    │  it finds — not a fixed sequence. This          │
    │  conditional, data-driven tool selection        │
    │  is what makes the system genuinely agentic.    │
    │                                                 │
    │  Tool 1: run_prediction()                       │
    │      Always called — need prediction first.     │
    │                                                 │
    │  Tool 2: get_shap_values()                      │
    │      Always called — per-prediction attribution │
    │                                                 │
    │  Tool 3: get_feature_context()                  │
    │      Called for top 3 importance features.      │
    │                                                 │
    │  Tool 4: check_sensitivity()                    │
    │      Called conditionally:                      │
    │        - Always for #1 importance feature       │
    │        - For conflicting features               │
    │        - For top SHAP features if borderline    │
    │                                                 │
    │  Output: structured investigation report        │
    └──────────────────┬──────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────┐
    │              Communicator Agent                  │
    │                                                 │
    │  Receives investigation report + session        │
    │  memory + aggregate stats.                      │
    │  Calls llm_reasoning.generate_llm_reasoning()   │
    │  to produce the final explanation.              │
    └─────────────────────────────────────────────────┘

Memory layers used:
    Working memory  — InvestigationState (LangGraph state dict)
                      Accumulates every tool call and finding.
    Session memory  — passed in from memory_store.get_session_summary()
    Long-term memory — passed in from memory_store.get_aggregate_stats()

LangSmith observability:
    Set in environment — no code changes needed:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=your_key
        LANGCHAIN_PROJECT=aquifer-groundwater
    Every graph invocation is traced: node names, tool I/O, latency,
    token usage. Visible at smith.langchain.com.

Imported by:
    main.py → from agent import run_investigation
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from reasoning import (
    FEATURE_INFO,
    FEATURE_IMPORTANCE,
    FEATURE_STATS,
    _level,
)
from llm_reasoning import generate_llm_reasoning
from train import normalise_shap

# ── Load model artifacts once at module import ────────────────────────────────
# Loading here (not inside tool functions) means disk reads happen once
# at server startup, not on every request.
BASE      = Path(__file__).parent
MODEL_DIR = BASE / "model"

_model     = joblib.load(MODEL_DIR / "rf_model.joblib")
_explainer = joblib.load(MODEL_DIR / "shap_explainer.joblib")
FEATURE_COLS = json.loads(
    (MODEL_DIR / "feature_columns.json").read_text()
)


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE — Working Memory Layer
# ══════════════════════════════════════════════════════════════════════════════

class InvestigationState(TypedDict):
    """
    Shared state dict passed between all LangGraph nodes.
    This is the working memory of the agent system.

    LangGraph passes this from node to node. Each node reads what it
    needs and returns a PARTIAL dict of updates. LangGraph merges
    updates into the full state automatically.

    Fields with Annotated[list, operator.add] use LangGraph's reducer:
    new items returned by any node are APPENDED to the existing list,
    not replaced. This is how investigation_notes builds a cumulative
    trace across both agent nodes without explicit read-before-write.
    """

    # ── Inputs ────────────────────────────────────────────────────────────
    feature_values:  dict         # 10 feature values from the user form
    session_id:      str          # FastAPI session uuid
    session_summary: dict | None  # Layer 2 memory from memory_store
    agg_stats:       dict | None  # Layer 3 memory from memory_store
    place_name:      str  | None  # reverse-geocoded location string

    # ── Working memory (filled during investigation) ───────────────────────
    prediction:            int   | None
    probability:           float | None
    shap_values:           dict  | None
    sensitivity_results:   list
    flagged_conflicts:     list
    investigation_report:  str   | None

    # Annotated with operator.add: new list items are appended, not replaced
    investigation_notes: Annotated[list, operator.add]

    # ── Outputs (set by Communicator Agent) ───────────────────────────────
    final_reasoning:  str  | None
    reasoning_source: str  | None
    reasoning_model:  str  | None


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# Called by the Investigator Agent node, not by LangGraph directly.
# Regular Python functions — no decorator needed in this architecture.
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(feature_values: dict) -> dict:
    """
    Tool 1: Run the trained RandomForest model.

    Always the first tool called — we need the prediction and probability
    before the agent can decide what else to investigate.

    Args:
        feature_values : {FEATURE_COL: float}

    Returns:
        {"prediction": int, "probability": float}
    """
    X     = pd.DataFrame([feature_values])[FEATURE_COLS]
    proba = float(_model.predict_proba(X)[0, 1])
    pred  = int(proba >= 0.5)
    return {"prediction": pred, "probability": round(proba, 4)}


def get_shap_values(feature_values: dict) -> dict:
    """
    Tool 2: Compute per-prediction SHAP values using TreeExplainer.

    SHAP answers: 'For THIS specific input, how much did each feature
    push the prediction toward class 1 (groundwater present)?'

    Positive SHAP = pushed toward groundwater present (class 1)
    Negative SHAP = pushed toward groundwater absent  (class 0)

    Uses normalise_shap() from train.py to handle output shape
    differences across shap library versions — same function,
    imported rather than duplicated.

    Args:
        feature_values : {FEATURE_COL: float}

    Returns:
        {FEATURE_COL: shap_value_float} for the positive class
    """
    X        = pd.DataFrame([feature_values])[FEATURE_COLS]
    raw_shap = _explainer.shap_values(X)

    # Always returns (1, n_features) after normalisation
    arr = normalise_shap(raw_shap, n_samples=1, n_features=len(FEATURE_COLS))

    return {
        col: round(float(arr[0, i]), 4)
        for i, col in enumerate(FEATURE_COLS)
    }


def get_feature_context(feature_name: str, value: float) -> dict:
    """
    Tool 3: Return distribution context for one feature value.

    Tells the agent whether the submitted value is low/medium/high
    relative to the training data, and what direction is favourable
    for groundwater. Used to detect conflicts with SHAP direction.

    Args:
        feature_name : one of the 10 FEATURE_COLS
        value        : user-submitted value

    Returns:
        Dict with distribution stats, level, importance rank,
        favorable direction, and plain-English description
    """
    if feature_name not in FEATURE_STATS:
        return {"error": f"Unknown feature: {feature_name}"}

    stats    = FEATURE_STATS[feature_name]
    info     = FEATURE_INFO.get(feature_name, {})
    level    = _level(feature_name, value)
    imp_keys = list(FEATURE_IMPORTANCE.keys())
    rank     = imp_keys.index(feature_name) + 1 if feature_name in imp_keys else None

    return {
        "feature":             feature_name,
        "label":               info.get("label", feature_name),
        "value":               value,
        "level":               level,
        "mean":                stats["mean"],
        "std":                 stats["std"],
        "min":                 stats["min"],
        "max":                 stats["max"],
        "importance_rank":     rank,
        "favorable_direction": info.get("favorable", "context"),
        "description":         info.get("description", ""),
    }


def check_sensitivity(
    feature_values: dict,
    feature_name:   str,
    delta:          int,
) -> dict:
    """
    Tool 4: Nudge one feature by delta class steps and rerun the model.

    Answers: 'How much would the prediction change if this feature
    were slightly different?' This is a conditional what-if analysis.

    Called selectively (not for every feature) to keep latency low.
    Each call runs the model twice (original + modified).

    Clipped to [min, max] so we never ask the model to predict outside
    its training range.

    Args:
        feature_values : original feature dict
        feature_name   : which feature to nudge
        delta          : class steps to move (typically +1 or -1)

    Returns:
        Dict with original_val, new_val, original_prob, new_prob, impact
    """
    if feature_name not in feature_values:
        return {"error": f"{feature_name} not in feature_values"}

    stats        = FEATURE_STATS[feature_name]
    original_val = feature_values[feature_name]
    new_val      = max(
        stats["min"],
        min(stats["max"], original_val + delta)
    )

    original_res = run_prediction(feature_values)
    modified_res = run_prediction(
        {**feature_values, feature_name: new_val}
    )

    return {
        "feature":       feature_name,
        "original_val":  original_val,
        "new_val":       new_val,
        "delta":         delta,
        "original_prob": original_res["probability"],
        "new_prob":      modified_res["probability"],
        "impact":        round(
                             modified_res["probability"]
                             - original_res["probability"],
                             4
                         ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Investigator Node
# ══════════════════════════════════════════════════════════════════════════════

def investigator_node(state: InvestigationState) -> dict:
    """
    Investigator Agent — plans and executes multi-step site analysis.

    This node embodies the agentic behaviour: it does not follow a
    fixed script. It reads findings at each step and decides what
    to investigate next. The path through the tools varies with input.

    Step-by-step decision logic:
        1. run_prediction       — always (need prediction first)
        2. get_shap_values      — always (per-prediction attribution)
        3. get_feature_context  — always for top 3 importance features
        4. Conflict detection   — derived from Steps 2+3, no tool call
        5. check_sensitivity    — CONDITIONAL:
             · always for #1 globally important feature
             · for features identified as conflicting (up to 2)
             · for top SHAP features if prediction is borderline (40-65%)
             · capped at 3 total checks to control latency
        6. Build investigation report for Communicator Agent

    Everything is logged to investigation_notes so the frontend's
    Agent Trace tab shows exactly what happened, and LangSmith
    captures the full trace if configured.

    Args:
        state : InvestigationState from LangGraph

    Returns:
        Partial state dict — LangGraph merges into full state
    """
    fv    = state["feature_values"]
    notes = []

    # ── Step 1: Predict ───────────────────────────────────────────────────
    pred_res    = run_prediction(fv)
    prediction  = pred_res["prediction"]
    probability = pred_res["probability"]
    notes.append(
        f"[Tool 1: run_prediction] → "
        f"prediction={prediction}, probability={probability:.4f}"
    )

    # ── Step 2: SHAP values ───────────────────────────────────────────────
    shap_vals    = get_shap_values(fv)
    top_shap_col = max(shap_vals, key=lambda k: abs(shap_vals[k]))
    notes.append(
        f"[Tool 2: get_shap_values] → "
        f"top driver: {top_shap_col} "
        f"(SHAP={shap_vals[top_shap_col]:+.3f})"
    )

    # ── Step 3: Feature context (top 3 by global importance) ──────────────
    top_cols    = list(FEATURE_IMPORTANCE.keys())[:3]
    feature_ctx = {
        col: get_feature_context(col, fv[col])
        for col in top_cols if col in fv
    }
    notes.append(
        f"[Tool 3: get_feature_context] → "
        f"checked: {list(feature_ctx.keys())}"
    )

    # ── Step 4: Conflict detection ────────────────────────────────────────
    # Conflict = SHAP direction disagrees with known favorable direction.
    # Example: RAINFALL is LOW (usually unfavorable) but SHAP is positive
    # (pushed toward groundwater present). Interesting — something else
    # might be compensating.
    conflicts = []
    for col, ctx in feature_ctx.items():
        favorable  = ctx.get("favorable_direction")
        level      = ctx.get("level")
        shap_sign  = shap_vals.get(col, 0)
        label      = ctx.get("label", col)

        if favorable == "high" and level == "low" and shap_sign > 0.05:
            conflicts.append(
                f"{label} is LOW (usually unfavorable) but SHAP is "
                f"positive — other features may be compensating"
            )
        elif favorable == "low" and level == "high" and shap_sign < -0.05:
            conflicts.append(
                f"{label} is HIGH (unfavorable direction) and SHAP "
                f"confirms it is reducing groundwater likelihood"
            )
        elif favorable == "high" and level == "high" and shap_sign < -0.05:
            conflicts.append(
                f"{label} is HIGH (usually favorable) but SHAP is "
                f"negative — unusual combination worth noting"
            )

    notes.append(
        f"[Conflict detection] → "
        f"{len(conflicts)} conflict(s)"
        + (f": {conflicts[0]}" if conflicts else "")
    )

    # ── Step 5: Conditional sensitivity checks ────────────────────────────
    is_borderline       = 0.38 <= probability <= 0.65
    features_to_probe   = []

    # Always probe the #1 globally important feature
    top1 = list(FEATURE_IMPORTANCE.keys())[0]
    features_to_probe.append(top1)

    # Add conflicting features (extract feature name from conflict string)
    for conflict_str in conflicts[:2]:
        for col in FEATURE_COLS:
            label = FEATURE_INFO.get(col, {}).get("label", col)
            if label in conflict_str and col not in features_to_probe:
                features_to_probe.append(col)
                break

    # If borderline, also probe top 2 SHAP features
    if is_borderline:
        top_shap_cols = sorted(
            shap_vals.keys(),
            key=lambda k: abs(shap_vals[k]),
            reverse=True,
        )[:2]
        for col in top_shap_cols:
            if col not in features_to_probe:
                features_to_probe.append(col)

    # Run sensitivity checks (capped at 3)
    sensitivity_results = []
    for col in features_to_probe[:3]:
        # Probe toward the "improving" direction
        delta  = 1 if prediction == 0 else -1
        result = check_sensitivity(fv, col, delta)
        sensitivity_results.append(result)
        impact = result["impact"] * 100
        notes.append(
            f"[Tool 4: check_sensitivity] "
            f"{col}: {result['original_val']}→{result['new_val']}, "
            f"impact {'↑' if result['impact']>0 else '↓'}"
            f"{abs(impact):.1f}pp"
        )

    # ── Step 6: Build investigation report ───────────────────────────────
    top_shap_sorted = sorted(
        shap_vals.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:4]
    shap_summary = ", ".join(
        f"{FEATURE_INFO.get(k,{}).get('label',k)}={v:+.3f}"
        for k, v in top_shap_sorted
    )

    sens_summary = ""
    for s in sensitivity_results:
        if abs(s.get("impact", 0)) >= 0.03:
            label = FEATURE_INFO.get(
                s["feature"], {}
            ).get("label", s["feature"])
            sens_summary += (
                f"If {label} changed {s['original_val']:.0f}→{s['new_val']:.0f}, "
                f"probability shifts {s['impact']*100:+.1f}pp "
                f"(to {s['new_prob']*100:.1f}%). "
            )

    conflict_summary = (
        " Conflicts: " + "; ".join(conflicts) if conflicts else ""
    )

    borderline_note = (
        " NOTE: prediction is borderline — small feature changes "
        "could flip the outcome."
        if is_borderline else ""
    )

    report = (
        f"PREDICTION: "
        f"{'Groundwater likely' if prediction==1 else 'Groundwater unlikely'} "
        f"(probability={probability*100:.1f}%).\n"
        f"TOP SHAP DRIVERS: {shap_summary}.\n"
        f"SENSITIVITY: "
        f"{sens_summary or 'Prediction robust to small feature changes.'}"
        f"{conflict_summary}"
        f"{borderline_note}"
    )

    notes.append(
        "[Investigator] Report built — handing off to Communicator Agent"
    )

    return {
        "prediction":           prediction,
        "probability":          probability,
        "shap_values":          shap_vals,
        "sensitivity_results":  sensitivity_results,
        "flagged_conflicts":    conflicts,
        "investigation_notes":  notes,
        "investigation_report": report,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Communicator Node
# ══════════════════════════════════════════════════════════════════════════════

def communicator_node(state: InvestigationState) -> dict:
    """
    Communicator Agent — generates the final user-facing explanation.

    Receives the full InvestigationState after the Investigator has run,
    including the structured report, SHAP values, session memory, and
    long-term aggregate stats.

    Calls llm_reasoning.generate_llm_reasoning() which tries Hugging Face
    first and falls back to the template engine automatically.

    Why two separate agents instead of one?
        Investigator = analytical (numbers, tool calls, pattern detection)
        Communicator = linguistic (clear prose for a non-expert reader)
        Mixing both concerns in one agent consistently produces worse
        output for both tasks. Separation of concerns applies to agents
        just as it does to software components.

    Args:
        state : full InvestigationState after investigator_node has run

    Returns:
        Partial state dict with final_reasoning, reasoning_source,
        reasoning_model, and one appended investigation_notes entry
    """
    result = generate_llm_reasoning(
        feature_values       = state["feature_values"],
        prediction           = state["prediction"],
        probability          = state["probability"],
        place_name           = state.get("place_name"),
        shap_values          = state["shap_values"],
        investigation_report = state["investigation_report"],
        session_summary      = state.get("session_summary"),
        agg_stats            = state.get("agg_stats"),
    )

    source_note = result["source"]
    if result.get("fallback_reason"):
        source_note += f" (fallback: {result['fallback_reason'][:80]})"

    return {
        "final_reasoning":  result["text"],
        "reasoning_source": result["source"],
        "reasoning_model":  result.get("model"),
        "investigation_notes": [
            f"[Communicator] Explanation generated via {source_note}"
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH — Graph assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Assemble and compile the two-node LangGraph.

    Graph structure (linear):
        START → investigator_node → communicator_node → END

    Linear because the Investigator always runs before the Communicator
    — there is no condition under which we skip the investigation step.

    Branching could be added later to short-circuit on very high
    confidence predictions (skip sensitivity checks when p > 0.95) or
    to route to different Communicator prompts based on confidence level.

    compile() returns a CompiledGraph callable:
        final_state = graph.invoke(initial_state)

    LangSmith traces this automatically when
    LANGCHAIN_TRACING_V2=true is set in the environment.
    """
    graph = StateGraph(InvestigationState)

    graph.add_node("investigator", investigator_node)
    graph.add_node("communicator", communicator_node)

    graph.set_entry_point("investigator")
    graph.add_edge("investigator", "communicator")
    graph.add_edge("communicator", END)

    return graph.compile()


# Compile once at module import — reused for all requests
_graph = build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_investigation(
    feature_values:  dict,
    session_id:      str         = "default",
    session_summary: dict | None = None,
    agg_stats:       dict | None = None,
    place_name:      str  | None = None,
) -> dict:
    """
    Run the full two-agent investigation for a set of feature values.

    This is the ONLY function main.py imports from this module.
    Tools, nodes, and the graph are all internal implementation details.

    Args:
        feature_values  : validated {FEATURE_COL: float} dict from main.py
        session_id      : FastAPI session uuid for memory tracking
        session_summary : from memory_store.get_session_summary()
        agg_stats       : from memory_store.get_aggregate_stats()
        place_name      : reverse-geocoded string or None

    Returns:
        Dict with all outputs the API endpoint needs:
            prediction           — 0 or 1
            probability          — float 0.0-1.0
            shap_values          — {feature: shap_float}
            sensitivity_results  — list of sensitivity dicts
            flagged_conflicts    — list of conflict strings
            investigation_notes  — full agent trace (list of strings)
            investigation_report — structured text (Investigator → Communicator)
            final_reasoning      — user-facing explanation (markdown)
            reasoning_source     — 'huggingface', 'template', 'template_fallback'
            reasoning_model      — HF model ID string or None
    """
    initial_state: InvestigationState = {
        "feature_values":       feature_values,
        "session_id":           session_id,
        "session_summary":      session_summary,
        "agg_stats":            agg_stats,
        "place_name":           place_name,
        # Everything below starts empty — filled by the agents
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
    }

    final_state = _graph.invoke(initial_state)

    return {
        "prediction":           final_state["prediction"],
        "probability":          final_state["probability"],
        "shap_values":          final_state["shap_values"],
        "sensitivity_results":  final_state["sensitivity_results"],
        "flagged_conflicts":    final_state["flagged_conflicts"],
        "investigation_notes":  final_state["investigation_notes"],
        "investigation_report": final_state["investigation_report"],
        "final_reasoning":      final_state["final_reasoning"],
        "reasoning_source":     final_state["reasoning_source"],
        "reasoning_model":      final_state["reasoning_model"],
    }