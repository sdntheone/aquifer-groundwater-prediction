"""
llm_reasoning.py
================
Calls the Hugging Face Inference API to generate the final user-facing
explanation for a groundwater prediction.

Called exclusively by communicator_node() in agent.py.
Never called directly by main.py.

Environment variables:
    HF_API_TOKEN
        Your Hugging Face token (hf_xxxx...).
        Get one free at huggingface.co → Settings → Access Tokens.
        The app works without this — it falls back to reasoning.py.

    HF_MODEL
        The model ID to use for text generation.
        Default: mistralai/Mistral-7B-Instruct-v0.3
        Any instruction-tuned model on the HF Inference API will work.

    HF_TIMEOUT_SECONDS
        Seconds to wait before giving up on the API call.
        Default: 20

Fallback behaviour:
    If HF_API_TOKEN is not set, or if the API call fails for ANY reason
    (network error, rate limit, empty response, unexpected JSON shape),
    this module falls back to reasoning.generate_reasoning() silently.
    The caller always gets a valid explanation — never an exception.

Prompt design notes:
    - Uses [INST]...[/INST] format expected by Mistral-Instruct
    - Features ranked by SHAP value (per-prediction) not global importance
    - Explicitly asks for combination/interaction reasoning
    - Provides session memory context for cross-site comparison
    - wait_for_model=True prevents 503 on cold model starts
"""

import os
import requests

from reasoning import (
    generate_reasoning,
    FEATURE_INFO,
    FEATURE_IMPORTANCE,
    _level,
)

# ── Config ────────────────────────────────────────────────────────────────────
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
HF_MODEL     = os.environ.get(
    "HF_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3",
).strip()
HF_URL     = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TIMEOUT = float(os.environ.get("HF_TIMEOUT_SECONDS", "20"))


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    feature_values:       dict,
    prediction:           int,
    probability:          float,
    place_name:           str  | None,
    shap_values:          dict | None,
    investigation_report: str  | None,
    session_summary:      dict | None,
    agg_stats:            dict | None,
) -> str:
    """
    Build the full instruction-following prompt.

    Feature lines are sorted by absolute SHAP value when available —
    this ranks by what actually mattered for THIS prediction, which is
    more informative than global importance rank.

    Args:
        feature_values       : {FEATURE_COL: value} from user
        prediction           : 0 or 1
        probability          : float 0.0–1.0
        place_name           : reverse-geocoded string or None
        shap_values          : {FEATURE_COL: shap_float} or None
        investigation_report : structured text from Investigator Agent
        session_summary      : from memory_store.get_session_summary()
        agg_stats            : from memory_store.get_aggregate_stats()

    Returns:
        Prompt string in [INST]...[/INST] format
    """

    # Sort features by SHAP importance (absolute) if available,
    # otherwise fall back to global importance rank
    if shap_values:
        cols = sorted(
            [c for c in feature_values if c in shap_values],
            key=lambda c: abs(shap_values[c]),
            reverse=True,
        )[:6]
    else:
        cols = list(FEATURE_IMPORTANCE.keys())[:6]

    feature_lines = []
    for col in cols:
        if col not in feature_values:
            continue
        val   = feature_values[col]
        level = _level(col, val)
        label = FEATURE_INFO.get(col, {}).get("label", col)
        imp   = FEATURE_IMPORTANCE.get(col, 0)

        shap_note = ""
        if shap_values and col in shap_values:
            sv  = shap_values[col]
            dir = "↑ toward present" if sv > 0 else "↓ toward absent"
            shap_note = f", SHAP={sv:+.3f} ({dir})"

        feature_lines.append(
            f"  • {label}: value={val:.0f} ({level}), "
            f"global importance={imp:.3f}{shap_note}"
        )

    # Build optional context sections
    location_section = (
        f"Location: {place_name}\n" if place_name else ""
    )

    investigation_section = ""
    if investigation_report:
        investigation_section = (
            f"\nInvestigator Agent findings:\n{investigation_report}\n"
        )

    session_section = ""
    if session_summary and session_summary.get("total_checked", 0) > 0:
        ss   = session_summary
        best = ss["highest_probability_site"]
        session_section = (
            f"\nSession context: user has checked {ss['total_checked']} "
            f"site(s) this session. Best site so far: {best['place']} "
            f"({best['probability']*100:.1f}% probability). "
            f"Compare this site briefly if the comparison adds insight.\n"
        )

    agg_section = ""
    if agg_stats and agg_stats.get("total_predictions", 0) > 5:
        ag = agg_stats
        agg_section = (
            f"\nHistorical context: across {ag['total_predictions']} "
            f"predictions in this system, {ag['groundwater_likely_pct']}% "
            f"were groundwater-likely with average probability "
            f"{ag['average_probability']*100:.1f}%.\n"
        )

    verdict = (
        "groundwater is likely present"
        if prediction == 1 else
        "groundwater is unlikely to be present"
    )

    prompt = (
        "[INST] You are a senior hydrogeologist explaining a machine learning "
        "prediction to a non-technical decision-maker. A Random Forest model "
        f"predicted that {verdict} "
        f"(probability = {probability*100:.1f}%).\n\n"
        f"{location_section}"
        f"{investigation_section}"
        f"{session_section}"
        f"{agg_section}"
        "Feature values ranked by their influence on THIS specific prediction:\n"
        + "\n".join(feature_lines)
        + "\n\nWrite a 5-7 sentence explanation following these rules:\n"
        "  1. Start directly with the conclusion — not with 'Based on' or 'The model'.\n"
        "  2. Explain how the TOP 2-3 features INTERACT and COMBINE to produce "
        "       this result — not a list of individual features.\n"
        "       Good example: 'High rainfall combined with gentle slopes and sparse "
        "       drainage gives water enough time to soak deep into the ground "
        "       rather than run off as surface flow.'\n"
        "  3. Reference specific feature values naturally in prose — "
        "       do not reproduce them as a bullet list.\n"
        "  4. Use plain hydrogeology concepts: infiltration, recharge, runoff, "
        "       permeability, water table. Avoid all other jargon.\n"
        "  5. If session context is available, add one sentence comparing "
        "       this site to the best previous site checked.\n"
        "  6. End with one concrete, actionable insight for the user.\n"
        "  7. Do not use bullet points, numbered lists, or headers.\n"
        "  8. Do not mention AI, models, or algorithms.\n"
        "[/INST]"
    )

    return prompt


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_llm_reasoning(
    feature_values:       dict,
    prediction:           int,
    probability:          float,
    place_name:           str  | None = None,
    shap_values:          dict | None = None,
    investigation_report: str  | None = None,
    session_summary:      dict | None = None,
    agg_stats:            dict | None = None,
) -> dict:
    """
    Generate the final user-facing explanation for a prediction.

    Tries the Hugging Face Inference API first.
    Falls back to reasoning.generate_reasoning() if:
        - HF_API_TOKEN is not set
        - API returns non-200 status
        - API returns {"error": ...} payload (e.g. rate limit)
        - Generated text is empty
        - Any network, timeout, or parse error

    Args:
        feature_values       : {FEATURE_COL: value} — user's inputs
        prediction           : 0 or 1
        probability          : model probability for class 1
        place_name           : reverse-geocoded place name or None
        shap_values          : per-prediction SHAP dict or None
        investigation_report : Investigator Agent's structured report
        session_summary      : from memory_store.get_session_summary()
        agg_stats            : from memory_store.get_aggregate_stats()

    Returns:
        dict with keys:
            text            — explanation string (markdown)
            source          — 'huggingface', 'template', 'template_fallback'
            model           — HF model ID or None
            fallback_reason — error message if fallback triggered, else None
    """

    # No token → use template directly (not a fallback, just the default path)
    if not HF_API_TOKEN:
        return {
            "text": generate_reasoning(
                feature_values, prediction, probability, shap_values
            ),
            "source":          "template",
            "model":           None,
            "fallback_reason": None,
        }

    # Build and send prompt
    prompt = _build_prompt(
        feature_values, prediction, probability, place_name,
        shap_values, investigation_report, session_summary, agg_stats,
    )

    try:
        resp = requests.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens":   320,
                    "temperature":      0.65,
                    "return_full_text": False,
                },
                "options": {
                    # If model is cold (unloaded), wait instead of 503 immediately
                    "wait_for_model": True,
                },
            },
            timeout=HF_TIMEOUT,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"HF API {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()

        # HF can return {"error": "..."} even on HTTP 200 (e.g. rate limit)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF API error payload: {data['error']}")

        # Success: [{"generated_text": "..."}]
        if (
            isinstance(data, list)
            and len(data) > 0
            and "generated_text" in data[0]
        ):
            text = data[0]["generated_text"].strip()
            if not text:
                raise RuntimeError("HF returned empty generated_text")
            return {
                "text":            text,
                "source":          "huggingface",
                "model":           HF_MODEL,
                "fallback_reason": None,
            }

        raise RuntimeError(f"Unexpected HF response shape: {str(data)[:200]}")

    except Exception as exc:
        fallback_text = generate_reasoning(
            feature_values, prediction, probability, shap_values
        )
        return {
            "text":            fallback_text,
            "source":          "template_fallback",
            "model":           None,
            "fallback_reason": str(exc)[:300],
        }