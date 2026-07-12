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
        "label": "Rainfall (Annual, classified)",
        "description": (
            "Annual rainfall classified into 5 ordinal classes. "
            "Class 1 = very low rainfall (< 700 mm/year, as seen in Jalaun/Datia districts). "
            "Class 3 = moderate (around 900–1000 mm, typical Bundelkhand average). "
            "Class 5 = very high (> 1100 mm/year, as seen in Sagar/Panna districts). "
            "Higher class = more water available to recharge the aquifer below."
        ),
        "favorable":    "high",
        "explain_high": (
            "higher rainfall class means more annual precipitation available "
            "to infiltrate and recharge the aquifer below"
        ),
        "explain_low": (
            "lower rainfall class means less annual precipitation, "
            "reducing groundwater recharge potential significantly"
        ),
    },
    "ELEVATION": {
        "label": "Elevation (classified)",
        "description": (
            "Land height above sea level, classified into 5 ordinal classes. "
            "In Bundelkhand, elevation ranges from ~150 m (river plains) to ~600 m "
            "(Vindhya Range hills). "
            "Class 1 = lowest elevation (valley floors, river plains — water collects here). "
            "Class 5 = highest elevation (hilltops — water drains away before it can soak in). "
            "Lower elevation classes favour groundwater accumulation."
        ),
        "favorable":    "low",
        "explain_high": (
            "higher elevation class means water drains away downslope "
            "rather than infiltrating locally"
        ),
        "explain_low": (
            "lower elevation class places this site in a valley or plain "
            "that collects and retains infiltrating water"
        ),
    },
    "LULC": {
        "label": "Land Use / Land Cover (classified)",
        "description": (
            "What covers the ground, classified into 5 ordinal classes by infiltration potential. "
            "Class 1 = water bodies or dense urban areas (minimal infiltration). "
            "Class 2 = barren/fallow land. "
            "Class 3 = agricultural land (moderate infiltration). "
            "Class 4 = scrubland/open forest. "
            "Class 5 = dense forest (maximum infiltration — forest floor absorbs "
            "far more rainfall than any other cover type)."
        ),
        "favorable":    "context",
        "explain_high": (
            "this land cover class is associated with better infiltration "
            "and groundwater recharge potential"
        ),
        "explain_low": (
            "this land cover class is associated with reduced infiltration "
            "and lower groundwater recharge"
        ),
    },
    "DRAINAGE": {
        "label": "Drainage Density (classified)",
        "description": (
            "How many stream channels exist per unit area, classified into 5 ordinal classes. "
            "In Bundelkhand studies, drainage density ranges from ~0.5 to ~2.3 km/km². "
            "Class 1 = very high drainage density (streams everywhere — water is shed "
            "quickly as runoff, leaving little time to soak in). "
            "Class 5 = very low drainage density (sparse streams — water lingers "
            "longer and has more time to percolate underground). "
            "Lower drainage density classes favour groundwater recharge."
        ),
        "favorable":    "low",
        "explain_high": (
            "high drainage density class means water is efficiently channelled "
            "away as surface runoff before it can infiltrate"
        ),
        "explain_low": (
            "low drainage density class means water lingers on the surface "
            "longer, giving it more time to percolate underground"
        ),
    },
    "NDVI": {
        "label": "Vegetation Index — NDVI (classified)",
        "description": (
            "Satellite-derived measure of vegetation health and density, "
            "classified into 5 ordinal classes. "
            "Raw NDVI ranges from -1 to +1: negative values = water/clouds/snow; "
            "0–0.1 = bare rock or soil; 0.2–0.5 = sparse shrubs/grassland; "
            "0.5–0.9 = dense forest/healthy crops. "
            "Class 1 = very low NDVI (bare, degraded land). "
            "Class 5 = very high NDVI (dense healthy vegetation — strong indicator "
            "of subsurface moisture availability)."
        ),
        "favorable":    "high",
        "explain_high": (
            "high NDVI class indicates dense, healthy vegetation — "
            "a reliable proxy for good subsurface moisture and recharge conditions"
        ),
        "explain_low": (
            "low NDVI class indicates sparse or stressed vegetation, "
            "suggesting limited subsurface moisture"
        ),
    },
    "LITHOLOGY": {
        "label": "Lithology — Rock/Soil Type (classified)",
        "description": (
            "The type of rock or soil beneath the surface, classified by permeability. "
            "Bundelkhand is dominated by Precambrian granite and gneiss (hard rock, "
            "low primary porosity) with alluvial patches along rivers. "
            "Class 1 = dense crystalline rock (granite/quartzite) — very low permeability. "
            "Class 3 = weathered/fractured rock — moderate permeability. "
            "Class 5 = alluvium/sand/gravel — highest permeability and water storage. "
            "This is often the single most important factor for groundwater."
        ),
        "favorable":    "context",
        "explain_high": (
            "higher lithology class indicates more permeable or fractured rock, "
            "allowing water to move through and accumulate underground"
        ),
        "explain_low": (
            "lower lithology class indicates dense impermeable rock that "
            "blocks water movement and limits underground storage"
        ),
    },
    "SLOPE": {
        "label": "Slope (classified)",
        "description": (
            "Terrain steepness classified into 5 ordinal classes. "
            "In groundwater studies, slope is typically measured in degrees: "
            "Class 1 = nearly flat (0–2°, excellent infiltration — water has maximum "
            "time to soak in). "
            "Class 2 = gentle (2–5°). "
            "Class 3 = moderate (5–15°). "
            "Class 4 = steep (15–30°). "
            "Class 5 = very steep (> 30°, water runs off almost immediately). "
            "Flat terrain always favours groundwater recharge over steep terrain."
        ),
        "favorable":    "low",
        "explain_high": (
            "high slope class means steep terrain where rainwater runs off "
            "quickly before it can infiltrate the ground"
        ),
        "explain_low": (
            "low slope class means nearly flat terrain where water moves slowly "
            "and has maximum time to percolate underground"
        ),
    },
    "CURVATURE": {
        "label": "Surface Curvature (classified)",
        "description": (
            "Whether the land surface curves inward or outward at this point, "
            "classified into 5 ordinal classes. "
            "Raw curvature in GIS typically ranges from around -1.5 to +2.2 (unitless). "
            "Negative values = concave surface (bowl-shaped, collects water — "
            "favourable for recharge). "
            "Zero = flat surface. "
            "Positive values = convex surface (dome-shaped, sheds water — "
            "unfavourable for recharge). "
            "Class 1 = strongly concave; Class 5 = strongly convex in most classifications."
        ),
        "favorable":    "context",
        "explain_high": (
            "this curvature class indicates a convex or divergent surface "
            "that sheds water outward rather than collecting it"
        ),
        "explain_low": (
            "this curvature class indicates a concave or convergent surface "
            "that collects water from the surrounding terrain"
        ),
    },
    "SPI": {
        "label": "Stream Power Index — SPI (classified)",
        "description": (
            "A terrain index measuring the erosive power of surface water flow, "
            "classified into 5 ordinal classes. "
            "Raw SPI = ln(catchment area × tan(slope)) — typical values range "
            "from about -6 to +19 in GIS studies. "
            "Class 1 = low SPI (gentle terrain, limited flow power). "
            "Class 5 = very high SPI (active drainage channel with strong flow). "
            "High SPI zones mark natural drainage pathways that can also serve "
            "as recharge conduits into the subsurface."
        ),
        "favorable":    "context",
        "explain_high": (
            "high SPI class indicates an active drainage pathway with strong "
            "flow power — these zones can channel water into the subsurface"
        ),
        "explain_low": (
            "low SPI class indicates gentle terrain with limited surface flow "
            "concentration at this location"
        ),
    },
    "TWI": {
        "label": "Topographic Wetness Index — TWI (classified)",
        "description": (
            "Terrain index estimating how much water accumulates at a point, "
            "classified into 5 ordinal classes. "
            "Raw TWI = ln(catchment area / tan(slope)). "
            "Typical values: upper slopes ~5.5, mid slopes ~7.3, "
            "convergent lower slopes ~11.5, valley channels > 12. "
            "Class 1 = low TWI (ridges and upper slopes — dry zones). "
            "Class 5 = high TWI (valleys, floodplains — naturally wet zones "
            "where terrain geometry concentrates moisture). "
            "High TWI strongly favours groundwater recharge and presence."
        ),
        "favorable":    "high",
        "explain_high": (
            "high TWI class identifies this as a natural moisture-accumulation "
            "zone — terrain geometry concentrates water here, favouring recharge"
        ),
        "explain_low": (
            "low TWI class means the terrain geometry actively disperses water "
            "away from this point rather than concentrating it"
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