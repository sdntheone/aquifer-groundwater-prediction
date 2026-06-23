"""
llm_reasoning.py
Generates the groundwater-presence explanation using a hosted Hugging Face
text-generation model, with automatic fallback to the deterministic
template engine in `reasoning.py` if no HF token is configured, the
request fails, or it times out.

Setup:
    export HF_API_TOKEN="hf_xxx...."
    # optional, defaults to a small fast instruct model:
    export HF_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

If HF_API_TOKEN is not set, this module transparently falls back to
reasoning.generate_reasoning() so the app keeps working either way.
"""
import os
import json
import requests

from reasoning import generate_reasoning, FEATURE_IMPORTANCE, FEATURE_INFO, _level

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3").strip()
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TIMEOUT = float(os.environ.get("HF_TIMEOUT_SECONDS", "20"))


def _build_prompt(feature_values: dict, prediction: int, probability: float, place_name: str = None) -> str:
    ranked = list(FEATURE_IMPORTANCE.items())[:6]
    lines = []
    for col, importance in ranked:
        if col not in feature_values:
            continue
        val = feature_values[col]
        level = _level(col, val)
        label = FEATURE_INFO.get(col, {}).get("label", col)
        lines.append(f"- {label}: value={val} ({level} relative to the region), model importance={importance:.2f}")

    verdict = "groundwater is likely present" if prediction == 1 else "groundwater is unlikely to be present"
    location_line = f"Location context: {place_name}\n" if place_name else ""

    prompt = (
        "[INST] You are a hydrogeology assistant explaining a machine learning "
        "model's prediction to a non-technical user. A Random Forest model was "
        "trained on terrain, climate, and land-use features to predict groundwater "
        "well presence.\n\n"
        f"{location_line}"
        f"Model prediction: {verdict} (probability = {probability*100:.1f}%).\n\n"
        "The most influential features for this prediction, ranked by importance, are:\n"
        + "\n".join(lines) +
        "\n\nWrite a clear, concise explanation (5-7 sentences) in plain language for "
        "why this location got this prediction. Explicitly explain how the TOP 2-3 "
        "features interact and reinforce (or work against) each other together — "
        "groundwater presence is rarely caused by one factor alone, so describe the "
        "combined effect (e.g. 'high rainfall combined with low slope and low drainage "
        "density lets water linger and infiltrate, whereas high rainfall alone with steep "
        "slopes would mostly run off'). Reference the specific feature values given above "
        "and basic hydrogeology reasoning (recharge, infiltration, runoff, permeability, "
        "water table). Do not repeat the raw numbers verbatim in a list — weave them into "
        "natural prose. Be direct, avoid hedging filler, and do not mention that you "
        "are an AI model. [/INST]"
    )
    return prompt


def generate_llm_reasoning(feature_values: dict, prediction: int, probability: float, place_name: str = None) -> dict:
    """
    Returns {"text": str, "source": "huggingface" | "template", "model": str|None}
    """
    if not HF_API_TOKEN:
        text = generate_reasoning(feature_values, prediction, probability)
        return {"text": text, "source": "template", "model": None}

    prompt = _build_prompt(feature_values, prediction, probability, place_name)

    try:
        resp = requests.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 280,
                    "temperature": 0.6,
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            },
            timeout=HF_TIMEOUT,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"HF API returned {resp.status_code}: {resp.text[:300]}")

        data = resp.json()

        # Inference API can return a list of {"generated_text": ...} or a dict
        # with an "error" key (e.g. model loading, rate limit).
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"])

        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"].strip()
        else:
            raise RuntimeError(f"Unexpected HF response shape: {json.dumps(data)[:300]}")

        if not text:
            raise RuntimeError("Empty generation from HF model")

        return {"text": text, "source": "huggingface", "model": HF_MODEL}

    except Exception as e:
        # Always degrade gracefully — never let a flaky external API break
        # the core prediction feature.
        fallback_text = generate_reasoning(feature_values, prediction, probability)
        return {
            "text": fallback_text,
            "source": "template_fallback",
            "model": None,
            "fallback_reason": str(e)[:300],
        }
