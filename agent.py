"""
agent.py
========
Multi-agent LangGraph system for groundwater site investigation.

Architecture:
    ┌─────────────────────────────────────────────┐
    │            Investigator Agent                │
    │                                             │
    │  Has access to 4 tools. Decides which       │
    │  tools to call based on what it finds       │
    │  in the feature values — not a fixed        │
    │  sequence. This is what makes it agentic.   │
    │                                             │
    │  Tool 1: run_prediction()                   │
    │  Tool 2: get_shap_values()                  │
    │  Tool 3: get_feature_context()              │
    │  Tool 4: check_sensitivity()                │
    │                                             │
    │  Output: structured investigation report    │
    └──────────────────┬──────────────────────────┘
                       │ hands off report
                       ▼
    ┌─────────────────────────────────────────────┐
    │            Communicator Agent                │
    │                                             │
    │  Receives investigation report +            │
    │  session memory + long-term stats.          │
    │  Calls llm_reasoning.py to generate         │
    │  the final user-facing explanation.         │
    └─────────────────────────────────────────────┘

Memory layers:
    Working memory  — InvestigationState dict (LangGraph state)
                      tracks every tool call and finding within
                      a single investigation run
    Session memory  — passed in from memory_store.py
    Long-term memory — passed in from memory_store.py

Observability (LangSmith):
    Set these environment variables — no code changes needed:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=your_key_here
        LANGCHAIN_PROJECT=aquifer-groundwater
    Every graph invocation is then traced automatically including
    node names, inputs/outputs, latency, and token usage.

Usage:
    from agent import run_investigation

    result = run_investigation(
        feature_values  = {"ELEVATION": 2, "RAINFALL": 3, ...},
        session_id      = "abc-123",
        session_summary = {...},   # from memory_store
        agg_stats       = {...},   # from memory_store
        place_name      = "Jhansi, Uttar Pradesh",
    )
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
from train import normalise_shap   # reuse the same shape-normalisation logic

# ── Load model artifacts once at module import ────────────────────────────────
# Loaded here rather than in each tool call — avoids re-reading from disk
# on every request which would be slow.
BASE      = Path(__file__).parent
MODEL_DIR = BASE / "model"

_model     = joblib.load(MODEL_DIR / "rf_model.joblib")
_explainer = joblib.load(MODEL_DIR / "shap_explainer.joblib")
FEATURE_COLS = json.loads((MODEL_DIR / "feature_columns.json").read_text())


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE — Working Memory
# ══════════════════════════════════════════════════════════════════════════════

class InvestigationState(TypedDict):
    """
    The shared state object passed between all nodes in the LangGraph.
    This IS the working memory of the agent system.

    LangGraph passes this dict from node to node, each node reads what
    it needs and returns a partial dict of updates — LangGraph merges
    them automatically.

    Fields marked Annotated[list, operator.add] use LangGraph's
    reducer pattern: instead of replacing the list, new items are
    appended to it. This is how investigation_notes accumulates
    a trace of every step without each node needing to read the
    current list first.
    """

    # ── Inputs (set before graph starts) ─────────────────────────────────
    feature_values:  dict        # the 10 feature values from the user
    session_id:      str         # Flask session ID
    session_summary: dict | None # from memory_store (Layer 2)
    agg_stats:       dict | None # from memory_store (Layer 3)
    place_name:      str  | None # reverse-geocoded location name

    # ── Working memory (built up during investigation) ────────────────────
    prediction:            int   | None  # 0 or 1 from run_prediction tool
    probability:           float | None  # model confidence
    shap_values:           dict  | None  # {feature: shap_value} for this input
    sensitivity_results:   list          # list of sensitivity check dicts
    flagged_conflicts:     list          # features with conflicting signals
    investigation_report:  str   | None  # structured summary for Communicator

    # Annotated with operator.add — items are appended, not replaced
    # This gives us a full trace of every tool call made during the run
    investigation_notes: Annotated[list, operator.add]

    # ── Outputs (set by Communicator Agent) ───────────────────────────────
    final_reasoning:  str  | None  # user-facing explanation text
    reasoning_source: str  | None  # 'huggingface', 'template', etc.
    reasoning_model:  str  | None  # HF model ID or None


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — Called by the Investigator Agent
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(feature_values: dict) -> dict:
    """
    Tool 1: Run the trained RandomForest model on a feature dict.

    This is always the first tool called — we need the prediction
    and probability before deciding what else to investigate.

    Args:
        feature_values : dict of {FEATURE_COL: float}

    Returns:
        dict with keys: prediction (0 or 1), probability (float)
    """
    X     = pd.DataFrame([feature_values])[FEATURE_COLS]
    proba = float(_model.predict_proba(X)[0, 1])
    pred  = int(proba >= 0.5)
    return {"prediction": pred, "probability": round(proba, 4)}


def get_shap_values(feature_values: dict) -> dict:
    """
    Tool 2: Compute SHAP values for this specific prediction.

    SHAP (SHapley Additive exPlanations) answers the question:
    'For THIS particular set of feature values, how much did each
    feature push the prediction toward groundwater present vs absent?'

    Positive SHAP = pushed toward class 1 (groundwater present)
    Negative SHAP = pushed toward class 0 (groundwater absent)

    We use the normalise_shap() function from train.py to handle
    different output shapes across SHAP versions — same logic,
    imported rather than duplicated.

    Args:
        feature_values : dict of {FEATURE_COL: float}

    Returns:
        dict of {FEATURE_COL: shap_value} for the positive class
    """
    X        = pd.DataFrame([feature_values])[FEATURE_COLS]
    raw_shap = _explainer.shap_values(X)

    # normalise_shap always returns shape (n_samples, n_features)
    shap_array = normalise_shap(
        raw_shap,
        n_samples=1,
        n_features=len(FEATURE_COLS),
    )

    # shap_array[0] is the single row — zip with feature names
    return {
        col: round(float(shap_array[0, i]), 4)
        for i, col in enumerate(FEATURE_COLS)
    }


def get_feature_context(feature_name: str, value: float) -> dict:
    """
    Tool 3: Return where a feature value sits in the training distribution.

    Used by the Investigator Agent to understand whether a value is
    unusual relative to what the model was trained on, and to look up
    the feature's global importance rank and favorable direction.

    Args:
        feature_name : one of the 10 FEATURE_COLS
        value        : the user-submitted value for this feature

    Returns:
        dict with distribution stats, level, importance rank,
        favorable direction, and plain-English description
    """
    if feature_name not in FEATURE_STATS:
        return {"error": f"Unknown feature: {feature_name}"}

    stats    = FEATURE_STATS[feature_name]
    info     = FEATURE_INFO.get(feature_name, {})
    level    = _level(feature_name, value)

    # importance rank: 1 = most important globally
    imp_keys = list(FEATURE_IMPORTANCE.keys())
    rank     = imp_keys.index(feature_name) + 1 if feature_name in imp_keys else None

    return {
        "feature":            feature_name,
        "label":              info.get("label", feature_name),
        "value":              value,
        "level":              level,          # 'low', 'medium', 'high'
        "mean":               stats["mean"],
        "std":                stats["std"],
        "min":                stats["min"],
        "max":                stats["max"],
        "importance_rank":    rank,
        "favorable_direction":info.get("favorable", "context"),
        "description":        info.get("description", ""),
    }


def check_sensitivity(
    feature_values: dict,
    feature_name:   str,
    delta:          int,
) -> dict:
    """
    Tool 4: Nudge one feature by delta class steps and rerun the model.

    This answers the question: 'How much would the prediction change
    if this feature were slightly different?' — a what-if analysis
    that gives the user actionable insight.

    Example: if DRAINAGE is 4 (high, unfavorable) and we nudge it to 3,
    does the probability jump significantly? If yes, drainage density
    is a critical swing factor for this site.

    delta is typically +1 or -1 (one class step).
    The result is clipped to [min, max] from training data so we never
    ask the model to predict outside its training range.

    Args:
        feature_values : original user inputs
        feature_name   : which feature to nudge
        delta          : how many class steps to move (+1 or -1)

    Returns:
        dict with original_val, new_val, original_prob, new_prob,
        and impact (new_prob - original_prob)
    """
    if feature_name not in feature_values:
        return {"error": f"{feature_name} not in feature_values"}

    stats        = FEATURE_STATS[feature_name]
    original_val = feature_values[feature_name]

    # Clip so we don't go outside the trained range
    new_val = max(stats["min"], min(stats["max"], original_val + delta))

    # Run model with the nudged value
    modified     = {**feature_values, feature_name: new_val}
    original_res = run_prediction(feature_values)
    modified_res = run_prediction(modified)

    return {
        "feature":      feature_name,
        "original_val": original_val,
        "new_val":      new_val,
        "delta":        delta,
        "original_prob":original_res["probability"],
        "new_prob":     modified_res["probability"],
        "impact":       round(
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
    Investigator Agent — plans and executes a multi-step analysis.

    This node embodies the 'agentic' behaviour: it doesn't follow a
    fixed script. Instead, it reads what it finds at each step and
    decides what to investigate next.

    Decision logic:
        Step 1: Always call run_prediction (need this before anything else)
        Step 2: Always call get_shap_values (need per-prediction attributions)
        Step 3: Always call get_feature_context on top 3 importance features
        Step 4: Detect conflicts between SHAP direction and feature level
        Step 5: Conditionally call check_sensitivity —
                  - Always on the #1 global importance feature
                  - On conflicting features (to quantify the conflict)
                  - On top SHAP features if prediction is borderline (40-65%)
        Step 6: Build structured investigation report for Communicator Agent

    Everything is logged to investigation_notes so the full trace
    is visible in the UI (Agent Trace tab) and in LangSmith.

    Args:
        state : InvestigationState dict from LangGraph

    Returns:
        Partial state dict — LangGraph merges this into the full state
    """
    fv    = state["feature_values"]
    notes = []   # will become investigation_notes via Annotated[list, add]

    # ── Step 1: Run prediction ────────────────────────────────────────────
    pred_result = run_prediction(fv)
    prediction  = pred_result["prediction"]
    probability = pred_result["probability"]
    notes.append(
        f"[Tool 1: run_prediction] → "
        f"prediction={prediction}, probability={probability:.4f}"
    )

    # ── Step 2: SHAP values ───────────────────────────────────────────────
    shap_vals = get_shap_values(fv)
    top_shap_col = max(shap_vals, key=lambda k: abs(shap_vals[k]))
    notes.append(
        f"[Tool 2: get_shap_values] → "
        f"top SHAP driver: {top_shap_col} "
        f"(SHAP={shap_vals[top_shap_col]:+.3f})"
    )

    # ── Step 3: Feature context for top 3 by global importance ───────────
    top_importance_cols = list(FEATURE_IMPORTANCE.keys())[:3]
    feature_ctx = {
        col: get_feature_context(col, fv[col])
        for col in top_importance_cols
        if col in fv
    }
    notes.append(
        f"[Tool 3: get_feature_context] → "
        f"checked: {list(feature_ctx.keys())}"
    )

    # ── Step 4: Conflict detection ────────────────────────────────────────
    # A conflict means a feature's SHAP direction disagrees with its
    # known favorable direction — unusual pattern worth flagging.
    #
    # Example conflict: RAINFALL is LOW (unfavorable for groundwater)
    # but its SHAP value is positive (pushed prediction toward present).
    # This could mean nearby features are compensating, or the model
    # has learned a non-obvious interaction.
    conflicts = []
    for col, ctx in feature_ctx.items():
        favorable = ctx.get("favorable_direction")
        level     = ctx.get("level")
        shap_sign = shap_vals.get(col, 0)
        label     = ctx.get("label", col)

        # Feature that should be high is low but SHAP is still positive
        if favorable == "high" and level == "low" and shap_sign > 0.05:
            conflicts.append(
                f"{label} is LOW (usually unfavorable) but its SHAP is "
                f"positive — other features may be compensating"
            )
        # Feature that should be low is high and SHAP confirms it hurts
        elif favorable == "low" and level == "high" and shap_sign < -0.05:
            conflicts.append(
                f"{label} is HIGH (unfavorable direction) and SHAP "
                f"confirms it is reducing groundwater likelihood"
            )
        # Feature that should be high is high but SHAP is negative — unusual
        elif favorable == "high" and level == "high" and shap_sign < -0.05:
            conflicts.append(
                f"{label} is HIGH (usually favorable) but SHAP shows it "
                f"is negative — unusual combination worth noting"
            )

    notes.append(
        f"[Conflict detection] → "
        f"{len(conflicts)} conflict(s) found"
        + (f": {conflicts[0]}" if conflicts else "")
    )

    # ── Step 5: Conditional sensitivity checks ────────────────────────────
    # Build the list of features to probe. Cap at 3 total checks to keep
    # latency reasonable — each check reruns the model twice.
    is_borderline        = 0.38 <= probability <= 0.65
    features_to_probe    = []

    # Always probe the globally most important feature
    top_importance_col = list(FEATURE_IMPORTANCE.keys())[0]
    features_to_probe.append(top_importance_col)

    # Add conflicting features (up to 2)
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

    # Cap at 3, run sensitivity checks
    sensitivity_results = []
    for col in features_to_probe[:3]:
        # Probe in the direction that would improve the prediction
        # (toward 1 if currently 0, toward 0 if currently 1)
        delta  = 1 if prediction == 0 else -1
        result = check_sensitivity(fv, col, delta)
        sensitivity_results.append(result)

        impact_pct = result["impact"] * 100
        notes.append(
            f"[Tool 4: check_sensitivity] "
            f"{col}: {result['original_val']}→{result['new_val']}, "
            f"probability {'↑' if result['impact']>0 else '↓'}"
            f"{abs(impact_pct):.1f}pp"
        )

    # ── Step 6: Build investigation report ───────────────────────────────
    # This report is what the Communicator Agent receives.
    # It's structured text — not user-facing, but needs to be clear
    # enough for the LLM to extract the key insights from it.
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
        if abs(s.get("impact", 0)) >= 0.03:   # only mention meaningful changes
            label = FEATURE_INFO.get(s["feature"], {}).get("label", s["feature"])
            sens_summary += (
                f"If {label} changed from class {s['original_val']:.0f} "
                f"to {s['new_val']:.0f}, probability would shift "
                f"{s['impact']*100:+.1f}pp "
                f"(to {s['new_prob']*100:.1f}%). "
            )

    conflict_summary = (
        " Conflicts: " + "; ".join(conflicts)
        if conflicts else ""
    )

    is_borderline_note = (
        " Note: prediction is borderline — small feature changes "
        "could flip the outcome."
        if is_borderline else ""
    )

    report = (
        f"PREDICTION: "
        f"{'Groundwater likely' if prediction==1 else 'Groundwater unlikely'} "
        f"(probability={probability*100:.1f}%).\n"
        f"TOP SHAP DRIVERS: {shap_summary}.\n"
        f"SENSITIVITY: "
        f"{sens_summary or 'Prediction is robust to small feature changes.'}"
        f"{conflict_summary}"
        f"{is_borderline_note}"
    )

    notes.append(f"[Investigator] Report built — handing off to Communicator")

    # Return partial state — LangGraph merges this into InvestigationState
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

    Receives the full InvestigationState including:
        - The structured investigation report from the Investigator Agent
        - SHAP values for this specific prediction
        - Session memory (previous sites the user checked)
        - Long-term aggregate stats from PostgreSQL

    Passes all of this to llm_reasoning.generate_llm_reasoning() which
    tries the Hugging Face API first and falls back to the template engine.

    This separation of concerns is intentional:
        Investigator = analytical reasoning (numbers, tool calls, patterns)
        Communicator = linguistic reasoning (clear prose for a non-expert)
    Mixing these in one agent produces worse output for both tasks.

    Args:
        state : full InvestigationState after Investigator has run

    Returns:
        Partial state dict with final_reasoning, reasoning_source,
        reasoning_model, and one more investigation_notes entry
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

    return {
        "final_reasoning":  result["text"],
        "reasoning_source": result["source"],
        "reasoning_model":  result.get("model"),
        "investigation_notes": [
            f"[Communicator] Explanation generated "
            f"via {result['source']}"
            + (
                f" — fallback reason: {result['fallback_reason']}"
                if result.get("fallback_reason") else ""
            )
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH — Build and compile the graph
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Assemble the two-node LangGraph.

    Graph structure:
        START → investigator_node → communicator_node → END

    This is a linear graph (no branching) because the Investigator
    always runs before the Communicator — there's no condition where
    we'd skip the investigation. Branching would be added if we wanted
    to short-circuit (e.g. skip sensitivity checks for very high-confidence
    predictions) — a natural next evolution of this system.

    compile() returns a CompiledGraph that behaves like a callable:
        final_state = graph.invoke(initial_state)

    LangSmith tracing is automatically applied to the compiled graph
    if LANGCHAIN_TRACING_V2=true is set in the environment.
    """
    graph = StateGraph(InvestigationState)

    # Register nodes
    graph.add_node("investigator",  investigator_node)
    graph.add_node("communicator",  communicator_node)

    # Define edges
    graph.set_entry_point("investigator")
    graph.add_edge("investigator", "communicator")
    graph.add_edge("communicator", END)

    return graph.compile()


# Compile once at module import — reused for every request
_graph = build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT — called by app.py
# ══════════════════════════════════════════════════════════════════════════════

def run_investigation(
    feature_values:  dict,
    session_id:      str        = "default",
    session_summary: dict | None = None,
    agg_stats:       dict | None = None,
    place_name:      str  | None = None,
) -> dict:
    """
    Run the full two-agent investigation for a set of feature values.

    This is the only function app.py needs to import from this module.
    Everything else (tools, nodes, graph) is internal implementation.

    Args:
        feature_values  : dict of {FEATURE_COL: float} — validated user inputs
        session_id      : Flask session uuid for working memory tracking
        session_summary : output of memory_store.get_session_summary()
        agg_stats       : output of memory_store.get_aggregate_stats()
        place_name      : reverse-geocoded place name or None

    Returns:
        dict with all investigation outputs:
            prediction           — 0 or 1
            probability          — float
            shap_values          — {feature: shap_value}
            sensitivity_results  — list of sensitivity check dicts
            flagged_conflicts    — list of conflict description strings
            investigation_notes  — full agent trace (list of strings)
            investigation_report — structured report (Investigator → Communicator)
            final_reasoning      — user-facing explanation text
            reasoning_source     — 'huggingface', 'template', or 'template_fallback'
            reasoning_model      — HF model ID or None
    """
    # Build the initial state — this is where working memory starts
    initial_state: InvestigationState = {
        "feature_values":       feature_values,
        "session_id":           session_id,
        "session_summary":      session_summary,
        "agg_stats":            agg_stats,
        "place_name":           place_name,

        # These are all None/empty at the start — filled by the agents
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

    # invoke() runs the full graph synchronously and returns final state
    final_state = _graph.invoke(initial_state)

    # Return only what app.py needs — don't expose internal state keys
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