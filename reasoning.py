"""
reasoning.py
============
Two responsibilities:

    1. FEATURE_INFO dict
       Metadata for all 10 input features.
       Each entry has: label, description, favorable direction,
       and per-direction explanation strings.

       Imported by:
           main.py          → /api/feature_info endpoint (glossary on Page 1)
           llm_reasoning.py → injected into the LLM prompt as context
           agent.py         → get_feature_context() tool

    2. generate_reasoning()
       Deterministic template-based explanation engine.
       Returns a markdown-formatted string.

       Used as fallback when:
           - HF_API_TOKEN is not set
           - The Hugging Face API call fails for any reason

       Accepts optional shap_values so bullet points reference per-prediction
       SHAP attributions rather than just global importance rank.

Why keep this alongside an LLM?
    The LLM can fail — rate limit, network error, model cold-start timeout.
    This engine runs entirely locally with zero external dependencies.
    The app degrades gracefully to this instead of returning an error.
    Users don't know or care which path ran — they get an explanation either way.
"""

import json
from pathlib import Path

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE      = Path(__file__).parent
MODEL_DIR = BASE / "model"

FEATURE_IMPORTANCE = json.loads(
    (MODEL_DIR / "feature_importance.json").read_text()
)
FEATURE_STATS = json.loads(
    (MODEL_DIR / "feature_stats.json").read_text()
)


# ── Feature metadata ──────────────────────────────────────────────────────────

FEATURE_INFO = {
    "RAINFALL": {
        "label": "Rainfall",
        "description": (
            "How much precipitation this area typically receives. "
            "More rainfall means more water available to soak into the ground "
            "and recharge the aquifer below. Think of it as how often and "
            "how heavily it rains across the year at this location."
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
            "draining in from surrounding areas"
        ),
    },
    "LULC": {
        "label": "Land Use / Land Cover",
        "description": (
            "What covers the ground at this location — forest, farmland, "
            "built-up area, bare soil, water body, etc. This controls how "
            "easily rainwater soaks into the ground versus running off the "
            "surface. A forest floor absorbs far more than a paved road."
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
            "Lush vegetation signals accessible moisture below the surface "
            "— plants need water to grow, so healthy vegetation is a proxy "
            "for subsurface moisture availability."
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
            "The type of rock or soil beneath the surface. Some types — "
            "sand, gravel, fractured rock — let water pass through easily "
            "and store it. Others — dense clay, solid crystalline rock — "
            "block water movement entirely. This is often the single most "
            "important factor for groundwater presence."
        ),
        "favorable":    "context",
        "explain_high": (
            "this lithology class is relatively porous or fractured, "
            "supporting water movement and underground storage"
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
            "runoff — like water on a tilted glass versus a flat plate."
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
            "water). Concave areas concentrate flow from surrounding land; "
            "convex areas disperse it outward."
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
            "drainage channels. Low SPI means limited concentrated flow."
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
            "A terrain index estimating how much water accumulates at a "
            "given point, based on how flat it is and how much land drains "
            "into it from uphill. Higher TWI means the terrain geometry "
            "makes this a naturally wetter location — like the bottom of a "
            "shallow bowl collecting runoff from its surroundings."
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
    Classify a feature value as 'low', 'medium', or 'high' relative to
    the training data distribution using z-score thresholds of ±0.5.

    Thresholds are intentionally loose because all features are discretized
    integers (1–5) — tight thresholds would over-label every value as extreme.

    Args:
        col   : feature name (must exist in FEATURE_STATS)
        value : raw user-submitted value

    Returns:
        'low', 'medium', or 'high'
    """
    stats = FEATURE_STATS[col]
    mean  = stats["mean"]
    std   = stats["std"] or 1e-6   # guard against zero std
    z     = (value - mean) / std

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

    Output structure:
        1. Opening verdict (prediction label + probability + confidence word)
        2. Bullet points for top_n features by global importance
           Each bullet includes: label, level (low/med/high), importance rank,
           optional SHAP attribution, and a plain-English explanation string
        3. Combination reasoning paragraph (how top 2-3 features interact)

    This output is markdown-formatted. The frontend renders ** as bold.

    Args:
        feature_values : dict of {FEATURE_COL: float}
        prediction     : 0 or 1
        probability    : model probability for class 1 (0.0–1.0)
        shap_values    : optional {FEATURE_COL: float} from agent.py
                         if provided, bullet points include SHAP direction
        top_n          : how many features to cover in bullet points

    Returns:
        Markdown string ready for frontend rendering
    """
    ranked = list(FEATURE_IMPORTANCE.items())[:top_n]
    imp_keys = list(FEATURE_IMPORTANCE.keys())

    # ── Bullet points ─────────────────────────────────────────────────────
    bullets = []
    for col, importance in ranked:
        if col not in feature_values:
            continue

        val   = feature_values[col]
        level = _level(col, val)
        info  = FEATURE_INFO.get(col, {"label": col})
        rank  = imp_keys.index(col) + 1

        # SHAP note — only if SHAP values are available
        shap_note = ""
        if shap_values and col in shap_values:
            sv  = shap_values[col]
            dir = (
                "pushed toward groundwater present"
                if sv > 0 else
                "pushed toward groundwater absent"
            )
            shap_note = f" — SHAP={sv:+.3f} ({dir})"

        # Explanation string
        if level == "medium":
            explain = (
                "close to the regional average — roughly neutral effect "
                "on this prediction"
            )
        else:
            explain = info.get(f"explain_{level}", "")

        bullets.append(
            f"- **{info['label']}** is **{level}** "
            f"(value={val:.0f}, importance rank #{rank}{shap_note}): "
            f"{explain}."
        )

    # ── Confidence word ───────────────────────────────────────────────────
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

    # ── Assemble body ─────────────────────────────────────────────────────
    output = (
        f"Based on the trained Random Forest model, this location is {verdict} "
        f"(probability: {p*100:.1f}%, {confidence} confidence).\n\n"
        f"Key factors driving this decision, ranked by model importance:\n\n"
        + "\n".join(bullets)
    )

    # ── Combination reasoning ─────────────────────────────────────────────
    # Explains how the top 2-3 features interact — not just what each means.
    # This is the most useful insight for a non-technical user.
    combo_terms = []
    for col, _ in ranked[:3]:
        if col not in feature_values:
            continue
        level = _level(col, feature_values[col])
        label = FEATURE_INFO.get(col, {}).get("label", col)
        combo_terms.append(f"{label.lower()} ({level})")

    if len(combo_terms) >= 2:
        combo = ", ".join(combo_terms[:-1]) + " and " + combo_terms[-1]

        if prediction == 1:
            output += (
                f"\n\n**Why this combination matters:** {combo} occurring "
                f"together creates a compounding effect that favours recharge. "
                f"Groundwater accumulation requires both a water source "
                f"(sufficient rainfall reaching the surface) and the right "
                f"conditions for that water to infiltrate and stay underground "
                f"rather than run off or evaporate. When these top-ranked "
                f"factors align favourably at the same location, the pattern "
                f"closely matches confirmed wells in the training dataset."
            )
        else:
            output += (
                f"\n\n**Why this combination matters:** {combo} occurring "
                f"together tips the balance against groundwater accumulation. "
                f"Even if one factor alone were borderline, this combination "
                f"consistently favours surface runoff and evaporation over "
                f"infiltration and underground storage — a pattern the model "
                f"associates with absence of wells in the training data."
            )

    return output