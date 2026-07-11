"""
llm_reasoning.py
================
Calls the Hugging Face Inference API to generate the final user-facing
explanation for a groundwater prediction.

Called exclusively by the Communicator Agent node in agent.py.
Never called directly by app.py — always goes through the agent.

Environment variables:
    HF_API_TOKEN  — your Hugging Face token (hf_xxxx...)
                    get one free at huggingface.co → Settings → Access Tokens
    HF_MODEL      — model to use for generation
                    default: mistralai/Mistral-7B-Instruct-v0.3
                    any instruction-tuned model on HF Inference API works
    HF_TIMEOUT    — seconds to wait before giving up on the API call
                    default: 20

Fallback behaviour:
    If HF_API_TOKEN is not set, or the API call fails for any reason
    (rate limit, network error, model loading, empty response), this
    module transparently falls back to reasoning.generate_reasoning().
    The app never returns an error to the user because of a failed LLM call.

What makes a good prompt for this task:
    The model needs to explain a prediction, not just describe features.
    So the prompt:
      - Gives the model a clear role (hydrogeology expert, not generic assistant)
      - Provides SHAP values so it knows which features actually mattered
        for THIS prediction (not just global importance)
      - Provides the Investigator Agent's report so it knows what was probed
      - Provides session context so it can compare to previous sites
      - Explicitly asks for combination reasoning — the interaction between
        features, not a list of individual feature descriptions
      - Constrains output length and format to keep the UI clean
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
    "mistralai/Mistral-7B-Instruct-v0.3"
).strip()
HF_URL     = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TIMEOUT = float(os.environ.get("HF_TIMEOUT_SECONDS", "20"))


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    feature_values:      dict,
    prediction:          int,
    probability:         float,
    place_name:          str  | None,
    shap_values:         dict | None,
    investigation_report:str  | None,
    session_summary:     dict | None,
    agg_stats:           dict | None,
) -> str:
    """
    Build the full instruction-following prompt for the LLM.

    The prompt follows the [INST] ... [/INST] format expected by
    Mistral-Instruct and many other instruction-tuned models on HF.

    Args:
        feature_values       : dict of {FEATURE_COL: value} — user inputs
        prediction           : 0 or 1
        probability          : model probability for class 1
        place_name           : reverse-geocoded place name or None
        shap_values          : per-prediction SHAP dict or None
        investigation_report : structured report from Investigator Agent or None
        session_summary      : session memory summary dict or None
        agg_stats            : long-term aggregate stats dict or None

    Returns:
        Formatted prompt string ready to send to the HF Inference API.
    """

    # ── Feature lines ─────────────────────────────────────────────────────
    # Rank features by absolute SHAP value if available (most informative
    # for THIS prediction), otherwise fall back to global importance rank.
    if shap_values:
        ranked_cols = sorted(
            [c for c in feature_values if c in shap_values],
            key=lambda c: abs(shap_values[c]),
            reverse=True,
        )[:6]
    else:
        ranked_cols = list(FEATURE_IMPORTANCE.keys())[:6]

    feature_lines = []
    for col in ranked_cols:
        if col not in feature_values:
            continue
        val        = feature_values[col]
        level      = _level(col, val)
        label      = FEATURE_INFO.get(col, {}).get("label", col)
        importance = FEATURE_IMPORTANCE.get(col, 0)

        shap_note = ""
        if shap_values and col in shap_values:
            sv        = shap_values[col]
            direction = "↑ toward present" if sv > 0 else "↓ toward absent"
            shap_note = f", SHAP={sv:+.3f} ({direction})"

        feature_lines.append(
            f"  • {label}: value={val:.0f} ({level}), "
            f"global importance={importance:.3f}{shap_note}"
        )

    # ── Context sections ──────────────────────────────────────────────────
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
        ss = session_summary
        best = ss["highest_probability_site"]
        session_section = (
            f"\nSession context (user has checked {ss['total_checked']} "
            f"site(s) this session):\n"
            f"  Best site so far: {best['place']} "
            f"({best['probability']*100:.1f}% probability)\n"
            f"  Compare the current site to their previous checks "
            f"if the comparison adds insight.\n"
        )

    aggregate_section = ""
    if agg_stats and agg_stats.get("total_predictions", 0) > 5:
        ag = agg_stats
        aggregate_section = (
            f"\nHistorical context (across {ag['total_predictions']} "
            f"predictions in the system):\n"
            f"  {ag['groundwater_likely_pct']}% were groundwater-likely\n"
            f"  Average probability: {ag['average_probability']*100:.1f}%\n"
        )

    # ── Verdict line ──────────────────────────────────────────────────────
    verdict = (
        "groundwater is likely present"
        if prediction == 1 else
        "groundwater is unlikely to be present"
    )

    # ── Full prompt ───────────────────────────────────────────────────────
    prompt = (
        "[INST] You are a senior hydrogeologist explaining a machine learning "
        "prediction to a non-technical decision-maker. A Random Forest model "
        f"predicted that {verdict} at this location "
        f"(probability = {probability*100:.1f}%).\n\n"
        f"{location_section}"
        f"{investigation_section}"
        f"{session_section}"
        f"{aggregate_section}"
        "Feature values ranked by their influence on THIS prediction:\n"
        + "\n".join(feature_lines)
        + "\n\n"
        "Write a 5-7 sentence explanation following these rules:\n"
        "  1. Start directly with the conclusion — do not start with "
        "       'Based on' or 'The model shows'.\n"
        "  2. Explain how the TOP 2-3 features INTERACT and COMBINE "
        "       to produce this result — not a list of individual features.\n"
        "       Example of what we want: 'High rainfall combined with gentle "
        "       slopes and sparse drainage gives water enough time to soak "
        "       deep into the ground rather than run off.'\n"
        "  3. Reference specific feature values naturally in prose — "
        "       do not reproduce them as a bullet list.\n"
        "  4. Use plain hydrogeology concepts: infiltration, recharge, "
        "       runoff, permeability, water table. No jargon beyond this.\n"
        "  5. If session context is provided, add one sentence comparing "
        "       this site to the best previous site the user checked.\n"
        "  6. End with one concrete, actionable insight for the user.\n"
        "  7. Do not mention that you are an AI or that a model was used.\n"
        "  8. Do not use bullet points or numbered lists in your response.\n"
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
        - The API returns a non-200 status
        - The API returns an error payload
        - The generated text is empty
        - Any network or timeout error occurs

    Args:
        feature_values       : dict of {FEATURE_COL: value}
        prediction           : 0 or 1
        probability          : model probability for class 1
        place_name           : reverse-geocoded place name or None
        shap_values          : per-prediction SHAP dict from agent or None
        investigation_report : structured report from Investigator Agent
        session_summary      : summary from memory_store.get_session_summary()
        agg_stats            : stats from memory_store.get_aggregate_stats()

    Returns:
        dict with keys:
            text            — the explanation string (markdown supported)
            source          — 'huggingface', 'template', or 'template_fallback'
            model           — HF model ID or None
            fallback_reason — error message if fallback was triggered, else None
    """

    # ── No token configured — use template directly ───────────────────────
    if not HF_API_TOKEN:
        return {
            "text":            generate_reasoning(
                                   feature_values, prediction,
                                   probability, shap_values
                               ),
            "source":          "template",
            "model":           None,
            "fallback_reason": None,
        }

    # ── Try Hugging Face API ──────────────────────────────────────────────
    prompt = _build_prompt(
        feature_values, prediction, probability, place_name,
        shap_values, investigation_report, session_summary, agg_stats,
    )

    try:
        response = requests.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens":  320,
                    "temperature":     0.65,
                    "return_full_text": False,   # return only the generated part
                },
                "options": {
                    # wait_for_model: if the model is cold (not loaded),
                    # wait instead of returning a 503 immediately
                    "wait_for_model": True,
                },
            },
            timeout=HF_TIMEOUT,
        )

        # Non-200 response — treat as failure
        if response.status_code != 200:
            raise RuntimeError(
                f"HF API returned {response.status_code}: "
                f"{response.text[:200]}"
            )

        data = response.json()

        # API can return {"error": "..."} even on HTTP 200 (e.g. rate limit)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF API error: {data['error']}")

        # Successful response: list of {"generated_text": "..."}
        if (
            isinstance(data, list)
            and len(data) > 0
            and "generated_text" in data[0]
        ):
            text = data[0]["generated_text"].strip()
            if not text:
                raise RuntimeError("HF returned an empty generated_text")

            return {
                "text":            text,
                "source":          "huggingface",
                "model":           HF_MODEL,
                "fallback_reason": None,
            }

        # Unexpected response shape
        raise RuntimeError(
            f"Unexpected HF response structure: {str(data)[:200]}"
        )

    except Exception as exc:
        # ── Fallback to template ──────────────────────────────────────────
        # Log the reason so it's visible in server logs / LangSmith trace,
        # but never surface the error to the user.
        fallback_text = generate_reasoning(
            feature_values, prediction, probability, shap_values
        )
        return {
            "text":            fallback_text,
            "source":          "template_fallback",
            "model":           None,
            "fallback_reason": str(exc)[:300],
        }