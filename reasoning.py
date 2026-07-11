"""
reasoning.py
============
Two responsibilities:

    1. FEATURE_INFO dict
       Metadata for all 10 input features — label, plain-English description,
       which direction is favourable for groundwater, and per-direction
       explanation strings used in bullet points.
       Imported by:
           app.py          → /api/feature_info endpoint (glossary)
           llm_reasoning.py → injected into LLM prompt as context
           agent.py         → get_feature_context tool

    2. generate_reasoning()
       Deterministic template-based explanation engine.
       Used as the fallback when no HF_API_TOKEN is configured,
       or when the Hugging Face API call fails.
       Accepts optional SHAP values so bullet points can reference
       per-prediction feature attributions rather than just global importance.

Why keep this alongside the LLM?
    The LLM can fail (rate limit, network, token expired).
    This engine never fails — it runs entirely locally with no dependencies
    beyond what's already loaded. The app degrades gracefully instead of
    returning an error to the user.
"""

import json
from pathlib import Path

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE      = Path(__file__).parent
MODEL_DIR = BASE / "model"

FEATURE_IMPORTANCE = json.loads((MODEL_DIR / "feature_importance.json").read_text())
FEATURE_STATS      = json.loads((MODEL_DIR / "feature_stats.json").read_text())


# ── Feature metadata ──────────────────────────────────────────────────────────
# Each entry contains:
#   label       — short display name shown in the UI
#   description — plain English for the glossary (non-technical users)
#   favorable   — which direction favours groundwater: 'high', 'low', 'context'
#   explain_high / explain_low — used in reasoning bullet points

FEATURE_INFO = {
    "RAINFALL": {
        "label": "Rainfall",
        "description": (
            "How much precipitation this area typically receives. "
            "More rainfall means more water available to soak into the ground "
            "and recharge the aquifer below. Think of it as how often and how "
            "heavily it rains at this location across the year."
        ),
        "favorable":    "high",
        "explain_high": (
            "higher rainfall provides more water for aquifer recharge, "
            "raising the water table over time"
        ),
        "explain_low": (
            "lower rainfall means less water is available to percolate "
            "down and replenish the aquifer"
        ),
    },

    "ELEVATION": {
        "label": "Elevation",
        "description": (
            "Height of the land above sea level. Low-lying areas tend to "
            "collect water that drains down from surrounding higher ground. "
            "Higher areas shed water downslope before it has a chance to "
            "soak in — think of a hilltop versus a valley floor."
        ),
        "favorable":    "low",
        "explain_high": (
            "higher elevation causes water to drain away downslope "
            "rather than infiltrate locally"
        ),
        "explain_low": (
            "lower-lying terrain acts as a collection zone for water "
            "draining in from the surrounding area"
        ),
    },

    "LULC": {
        "label": "Land Use / Land Cover",
        "description": (
            "What covers the ground at this location — forest, farmland, "
            "built-up urban area, bare soil, water body, etc. This controls "
            "how easily rainwater soaks into the ground versus running off "
            "the surface. A forest floor absorbs far more than a paved road."
        ),
        "favorable":    "context",
        "explain_high": (
            "this land cover class is associated with reduced infiltration "
            "rates in this region"
        ),
        "explain_low": (
            "this land cover class supports better infiltration and "
            "groundwater recharge here"
        ),
    },

    "DRAINAGE": {
        "label": "Drainage Density",
        "description": (
            "How many streams and channels run through this area per unit "
            "of land. Densely-drained areas efficiently remove surface water "
            "as runoff, leaving little time for it to soak in. Sparsely "
            "drained areas let water linger and percolate underground."
        ),
        "favorable":    "low",
        "explain_high": (
            "dense drainage networks efficiently remove surface water, "
            "reducing the time available for infiltration"
        ),
        "explain_low": (
            "sparse drainage allows more water to linger on the surface "
            "long enough to percolate underground"
        ),
    },

    "NDVI": {
        "label": "Vegetation Index (NDVI)",
        "description": (
            "A satellite-derived measure of how green and healthy the "
            "vegetation is, on a scale from bare ground to dense forest. "
            "Lush vegetation often signals accessible moisture below the "
            "surface — plants need water to grow, so healthy vegetation "
            "is a proxy for subsurface moisture."
        ),
        "favorable":    "high",
        "explain_high": (
            "healthy dense vegetation indicates abundant subsurface "
            "moisture, suggesting good recharge conditions"
        ),
        "explain_low": (
            "sparse or stressed vegetation suggests limited "
            "subsurface moisture availability"
        ),
    },

    "LITHOLOGY": {
        "label": "Lithology (Rock / Soil Type)",
        "description": (
            "The type of rock or soil beneath the surface at this location. "
            "Some types — sand, gravel, or fractured rock — let water pass "
            "through easily and store it in pores and cracks. Others — "
            "dense clay or solid crystalline rock — block water movement "
            "entirely. This is often the single most important factor."
        ),
        "favorable":    "context",
        "explain_high": (
            "this lithology class is relatively porous or fractured, "
            "supporting water movement and storage underground"
        ),
        "explain_low": (
            "this lithology class has limited permeability, "
            "restricting groundwater accumulation"
        ),
    },

    "SLOPE": {
        "label": "Slope",
        "description": (
            "How steep the terrain is at this location. Gentle slopes give "
            "rainfall more time to soak into the ground before it flows "
            "away. Steep slopes cause water to rush off quickly as surface "
            "runoff with almost no time to infiltrate — like water on a "
            "tilted glass versus a flat plate."
        ),
        "favorable":    "low",
        "explain_high": (
            "steep slopes accelerate surface runoff, leaving little "
            "water available to infiltrate the subsurface"
        ),
        "explain_low": (
            "gentle slopes allow rainfall to linger long enough "
            "to percolate into the ground"
        ),
    },

    "CURVATURE": {
        "label": "Curvature",
        "description": (
            "Whether the land surface curves inward (concave — like a bowl, "
            "collecting water) or outward (convex — like a dome, shedding "
            "water) at this point. Concave areas concentrate flow from "
            "surrounding land; convex areas disperse it outward."
        ),
        "favorable":    "context",
        "explain_high": (
            "the surface curvature here causes water to converge, "
            "which can enhance local infiltration"
        ),
        "explain_low": (
            "the surface curvature causes water to diverge "
            "and spread away from this point"
        ),
    },

    "SPI": {
        "label": "Stream Power Index (SPI)",
        "description": (
            "A terrain-derived index measuring the erosive power of water "
            "flow at this point, combining slope steepness with how much "
            "land drains through here from upstream. High SPI marks active "
            "drainage channels. Low SPI means this is not a natural "
            "concentrated flow path."
        ),
        "favorable":    "context",
        "explain_high": (
            "high SPI indicates a natural drainage pathway that can "
            "channel recharge water into the subsurface"
        ),
        "explain_low": (
            "low SPI reflects limited concentrated surface flow "
            "at this location"
        ),
    },

    "TWI": {
        "label": "Topographic Wetness Index (TWI)",
        "description": (
            "A terrain index that estimates how much water tends to "
            "accumulate at a given point, based on how flat it is and "
            "how much land drains into it from uphill. Higher TWI means "
            "the terrain geometry naturally makes this a wetter location "
            "— like the bottom of a shallow bowl that collects runoff "
            "from its surroundings."
        ),
        "favorable":    "high",
        "explain_high": (
            "high TWI identifies this as a natural moisture-accumulation "
            "zone, favouring groundwater recharge"
        ),
        "explain_low": (
            "low TWI means the terrain geometry works against "
            "moisture accumulation at this point"
        ),
    },
}


# ── Utility ───────────────────────────────────────────────────────────────────

def _level(col: str, value: float) -> str:
    """
    Classify a feature value as 'low', 'medium', or 'high' relative
    to the training data distribution using z-score thresholds.

    Thresholds (±0.5 std) are intentionally loose — the features are
    discretized integers (1-5), so a ±0.5 std band avoids over-labelling
    every value as extreme.

    Args:
        col   : feature column name (must be in FEATURE_STATS)
        value : raw feature value submitted by the user

    Returns:
        'low', 'medium', or 'high'
    """
    stats       = FEATURE_STATS[col]
    mean        = stats["mean"]
    std         = stats["std"] or 1e-6   # guard against zero std
    z           = (value - mean) / std
    if z >= 0.5:  return "high"
    if z <= -0.5: return "low"
    return "medium"


# ── Template reasoning engine ─────────────────────────────────────────────────

def generate_reasoning(
    feature_values: dict,
    prediction:     int,
    probability:    float,
    shap_values:    dict | None = None,
    top_n:          int         = 4,
) -> str:
    """
    Generate a structured plain-English explanation for a prediction.

    Structure of the output:
        1. Opening verdict line (prediction + probability + confidence)
        2. Bullet points for top_n features (importance rank + level +
           SHAP direction if available + explanation string)
        3. Combination reasoning paragraph (how the top 2-3 features
           interact together to produce this outcome)
        4. Closing sentence summarising the overall picture

    Args:
        feature_values : dict of {FEATURE_COL: float} — user inputs
        prediction     : 0 or 1 from the model
        probability    : model probability for class 1 (0.0 – 1.0)
        shap_values    : optional dict of {FEATURE_COL: float} SHAP values
                         for this specific prediction (from agent.py)
        top_n          : how many features to include in bullet points

    Returns:
        Markdown-formatted string (bold via ** supported by the frontend)
    """
    ranked = list(FEATURE_IMPORTANCE.items())[:top_n]

    # ── Build bullet points ───────────────────────────────────────────────
    bullet_points = []
    for col, importance in ranked:
        if col not in feature_values:
            continue

        val   = feature_values[col]
        level = _level(col, val)
        info  = FEATURE_INFO.get(col, {"label": col})

        # SHAP note — only included if SHAP values were computed
        shap_note = ""
        if shap_values and col in shap_values:
            sv        = shap_values[col]
            direction = (
                "pushed this prediction toward groundwater present"
                if sv > 0 else
                "pushed this prediction toward groundwater absent"
            )
            shap_note = f" — SHAP={sv:+.3f} ({direction})"

        # Explanation string
        if level == "medium":
            explain = (
                "close to the regional average, so it has a "
                "roughly neutral effect on this prediction"
            )
        else:
            explain = info.get(f"explain_{level}", "")

        bullet_points.append(
            f"- **{info['label']}** is **{level}** "
            f"(value={val:.0f}, importance rank #{list(FEATURE_IMPORTANCE.keys()).index(col)+1}"
            f"{shap_note}): {explain}."
        )

    # ── Confidence label ──────────────────────────────────────────────────
    p = probability
    confidence = (
        "very high" if p > 0.85 or p < 0.15 else
        "high"      if p > 0.70 or p < 0.30 else
        "moderate"
    )

    verdict = (
        "**likely to have groundwater**"
        if prediction == 1 else
        "**unlikely to have significant groundwater**"
    )

    # ── Assemble output ───────────────────────────────────────────────────
    summary = (
        f"Based on the trained Random Forest model, this location is {verdict} "
        f"(probability: {p*100:.1f}%, {confidence} confidence).\n\n"
        f"The following factors drove this decision, ranked by how much "
        f"the model relies on them overall:\n\n"
        + "\n".join(bullet_points)
    )

    # ── Combination reasoning ─────────────────────────────────────────────
    # Groundwater presence is almost never caused by one factor alone.
    # This paragraph explains how the top 2-3 factors work together —
    # which is the most useful insight for a non-technical user.
    combo_terms = []
    for col, _ in ranked[:3]:
        if col not in feature_values:
            continue
        level = _level(col, feature_values[col])
        label = FEATURE_INFO.get(col, {}).get("label", col)
        combo_terms.append(f"{label.lower()} ({level})")

    if len(combo_terms) >= 2:
        combo = (
            ", ".join(combo_terms[:-1]) + " and " + combo_terms[-1]
        )
        if prediction == 1:
            summary += (
                f"\n\n**Why this combination matters:** {combo} occurring "
                f"together creates a compounding effect that favours "
                f"recharge. Groundwater accumulation requires both a water "
                f"source (sufficient rainfall reaching the surface) and the "
                f"right conditions for that water to infiltrate and stay "
                f"underground rather than run off or evaporate. When the "
                f"top-ranked factors align favourably at the same location, "
                f"the pattern closely matches confirmed wells in the "
                f"training dataset."
            )
        else:
            summary += (
                f"\n\n**Why this combination matters:** {combo} occurring "
                f"together tips the balance against groundwater accumulation. "
                f"Even if one factor alone were borderline, the combination "
                f"consistently favours surface runoff and evaporation over "
                f"infiltration and underground storage — a pattern the model "
                f"associates with absence of wells in the training data."
            )

    return summary