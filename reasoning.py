"""
reasoning.py
Generates a human-readable, LLM-style explanation for why a given location
is (or isn't) likely to have groundwater, based on the trained model's
feature importances and how the location's feature values compare to the
overall dataset distribution.

This is a deterministic, template-based "reasoning engine" rather than a
call to a hosted LLM, so the deployed app has zero external API
dependency / API-key requirement. If you want to swap in a real LLM call
(OpenAI/Anthropic) see `llm_reasoning_stub()` at the bottom for where to
plug it in.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"

FEATURE_IMPORTANCE = json.loads((MODEL_DIR / "feature_importance.json").read_text())
FEATURE_STATS = json.loads((MODEL_DIR / "feature_stats.json").read_text())

# Human-friendly descriptions of what each factor means hydrologically,
# and which "direction" tends to favor groundwater presence.
FEATURE_INFO = {
    "RAINFALL": {
        "label": "Rainfall",
        "description": "How much precipitation this area typically receives. More rainfall means more water available to soak into the ground.",
        "favorable": "high",
        "explain_high": "higher rainfall recharges aquifers and raises the water table",
        "explain_low": "lower rainfall provides less recharge to the aquifer",
    },
    "ELEVATION": {
        "label": "Elevation",
        "description": "Height of the land above sea level. Low-lying areas tend to collect water that drains down from higher ground nearby.",
        "favorable": "low",
        "explain_high": "higher elevation usually means water drains away rather than accumulating",
        "explain_low": "lower-lying terrain tends to collect and retain infiltrating water",
    },
    "LULC": {
        "label": "Land Use / Land Cover",
        "description": "What covers the ground here — forest, farmland, built-up area, bare soil, etc. This affects how easily rainwater soaks in versus running off.",
        "favorable": "context",
        "explain_high": "this land cover class is associated with reduced infiltration in this region",
        "explain_low": "this land cover class is associated with better infiltration in this region",
    },
    "DRAINAGE": {
        "label": "Drainage Density",
        "description": "How many streams/channels run through this area per unit of land. Densely-drained areas shed rainwater quickly via surface flow instead of letting it soak in.",
        "favorable": "low",
        "explain_high": "dense drainage networks shed water quickly rather than letting it percolate",
        "explain_low": "sparse drainage density allows more water to infiltrate underground",
    },
    "NDVI": {
        "label": "Vegetation Index (NDVI)",
        "description": "A satellite-derived measure of how green/healthy the vegetation is. Healthy plant growth often signals there's accessible moisture below the surface.",
        "favorable": "high",
        "explain_high": "healthier vegetation often indicates available subsurface moisture",
        "explain_low": "sparse vegetation can indicate limited subsurface moisture",
    },
    "LITHOLOGY": {
        "label": "Lithology (rock/soil type)",
        "description": "The type of rock or soil beneath the surface. Some types (like sand or fractured rock) let water pass through and pool easily; others (like dense clay) block it.",
        "favorable": "context",
        "explain_high": "this lithology class tends to be more porous/permeable, aiding storage",
        "explain_low": "this lithology class tends to be less permeable, limiting storage",
    },
    "SLOPE": {
        "label": "Slope",
        "description": "How steep the ground is. Steep slopes cause rain to run off quickly; flatter ground gives water more time to seep underground.",
        "favorable": "low",
        "explain_high": "steeper slopes cause rapid runoff with little infiltration time",
        "explain_low": "gentle slopes allow rainfall more time to percolate into the ground",
    },
    "CURVATURE": {
        "label": "Curvature",
        "description": "Whether the land surface curves inward (concave, water-collecting) or outward (convex, water-shedding) at this point.",
        "favorable": "context",
        "explain_high": "this curvature pattern affects how water converges or diverges across the surface",
        "explain_low": "this curvature pattern affects how water converges or diverges across the surface",
    },
    "SPI": {
        "label": "Stream Power Index (SPI)",
        "description": "A measure of how much erosive force flowing water has at this point, based on slope and upstream catchment area. High SPI often marks natural drainage channels.",
        "favorable": "context",
        "explain_high": "high SPI indicates strong erosive flow that can carve recharge pathways",
        "explain_low": "low SPI indicates limited concentrated surface flow",
    },
    "TWI": {
        "label": "Topographic Wetness Index (TWI)",
        "description": "A terrain-based estimate of how much water tends to accumulate at this point, combining slope and upstream catchment area. Higher TWI = naturally wetter ground.",
        "favorable": "high",
        "explain_high": "a high TWI means the location is in a zone that tends to accumulate and retain moisture",
        "explain_low": "a low TWI means the location is less likely to accumulate moisture",
    },
}


def _level(col, value):
    """Classify a feature value as low/medium/high relative to the training distribution."""
    stats = FEATURE_STATS[col]
    mean, std = stats["mean"], stats["std"] or 1e-6
    z = (value - mean) / std
    if z >= 0.5:
        return "high"
    elif z <= -0.5:
        return "low"
    return "medium"


def generate_reasoning(feature_values: dict, prediction: int, probability: float, top_n: int = 4):
    """
    feature_values: dict of {FEATURE_COL: value}
    prediction: 0 or 1
    probability: model probability of class 1 (well present)
    """
    ranked = list(FEATURE_IMPORTANCE.items())[:top_n]

    bullet_points = []
    for col, importance in ranked:
        if col not in feature_values:
            continue
        val = feature_values[col]
        level = _level(col, val)
        info = FEATURE_INFO.get(col, {"label": col})
        if level == "medium":
            explain = "this factor is close to the regional average, so it has a neutral effect"
        else:
            explain = info.get(f"explain_{level}", "")
        bullet_points.append(
            f"- **{info['label']}** is **{level}** (value={val}, "
            f"importance weight={importance:.2f}): {explain}."
        )

    verdict = "likely to have groundwater" if prediction == 1 else "unlikely to have significant groundwater"
    confidence_word = (
        "very high" if probability > 0.85 or probability < 0.15 else
        "high" if probability > 0.7 or probability < 0.3 else
        "moderate"
    )

    summary = (
        f"Based on the trained Random Forest model, this location is **{verdict}** "
        f"(predicted probability of groundwater presence: {probability*100:.1f}%, "
        f"{confidence_word} confidence).\n\n"
        f"This conclusion is driven mainly by the following factors, ranked by how much "
        f"the model relies on them overall:\n\n" + "\n".join(bullet_points)
    )

    # explicit combination/interaction reasoning across the top 2-3 factors,
    # since groundwater presence is rarely driven by a single feature alone
    combo_terms = []
    for col, _ in ranked[:3]:
        if col not in feature_values:
            continue
        level = _level(col, feature_values[col])
        label = FEATURE_INFO.get(col, {}).get("label", col)
        combo_terms.append(f"{label.lower()} ({level})")

    if len(combo_terms) >= 2:
        combo_text = ", ".join(combo_terms[:-1]) + " and " + combo_terms[-1]
        if prediction == 1:
            combo_explain = (
                f"\n\nWhy this specific **combination** matters: it isn't any single factor "
                f"acting alone — it's {combo_text} occurring **together**. Recharge requires "
                f"both a water *source* (rainfall reaching the surface) and a way for that water "
                f"to *stay* rather than run off (favorable terrain such as low slope, low drainage "
                f"density, or high topographic wetness). When the top-ranked factors align this "
                f"way at the same location, the model sees the same pattern it learned from "
                f"confirmed wells in the training data."
            )
        else:
            combo_explain = (
                f"\n\nWhy this specific **combination** matters: {combo_text} occurring together "
                f"works against groundwater accumulation — even if one factor alone looked "
                f"borderline, the combination tips the balance toward water running off or "
                f"evaporating rather than infiltrating and accumulating underground."
            )
        summary += combo_explain

    if prediction == 1:
        summary += (
            "\n\nIn short, the combination of favorable recharge conditions (terrain that "
            "collects rather than sheds water, sufficient rainfall, and permeable surface "
            "conditions) makes groundwater accumulation plausible at this location."
        )
    else:
        summary += (
            "\n\nIn short, the terrain and climate conditions here are more consistent with "
            "runoff and limited infiltration than with groundwater accumulation, though "
            "localized aquifers can still exist at depth even when surface indicators are unfavorable."
        )

    return summary


def llm_reasoning_stub(feature_values: dict, prediction: int, probability: float):
    """
    Placeholder showing where you'd plug in a real hosted LLM call instead
    of the deterministic template above, e.g.:

        import anthropic
        client = anthropic.Anthropic()
        prompt = f"Explain in 3 sentences why a location with features " \\
                 f"{feature_values} was predicted as {prediction} " \\
                 f"(probability {probability}) for groundwater presence."
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    Left disabled by default so the deployed app has no external API
    dependency / cost / key requirement.
    """
    return generate_reasoning(feature_values, prediction, probability)
