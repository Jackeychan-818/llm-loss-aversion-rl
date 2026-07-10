#!/usr/bin/env python3
"""Build the fixed 50-good OOD evaluation suite.

Outputs:
  ood_new_goods_50.json       New goods master (50 goods, 10 new categories).
  ood_new_goods_50_test.json  All 1,225 pairs, 8 configurations per pair.
  OOD_NEW_GOODS_50_TABLE.md   Human-readable inventory and split summary.

The suite is evaluation-only. Do not use it for reward construction,
checkpoint selection, hyperparameter tuning, or training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent
SEED = 20260710
CONFIGS_PER_PAIR = 8


GOODS = {
    "Outdoor & Camping Equipment": {
        "Camping lantern": {
            "Illumination reach": ["tent-interior reach", "full-campsite reach", "wide-area trail reach"],
            "Weather sealing grade": ["dry-condition housing", "rain-resistant housing", "storm-sealed housing"],
        },
        "Hiking backpack": {
            "Trail load capacity": ["light day-hike load", "full-day trail load", "multi-day expedition load"],
            "Back support system": ["unpadded back panel", "contoured foam support", "ventilated suspension support"],
        },
        "Sleeping bag": {
            "Overnight temperature rating": ["mild-night rating", "cool-night rating", "subzero-night rating"],
            "Insulation fill grade": ["basic synthetic fill", "lofted synthetic fill", "premium down-alternative fill"],
        },
        "Portable camping stove": {
            "Burner output range": ["low-output simmer burner", "general-purpose camp burner", "high-output rapid-boil burner"],
            "Wind protection system": ["exposed flame ring", "partial wind collar", "integrated windscreen chamber"],
        },
        "Field binoculars": {
            "Distant image clarity": ["basic central clarity", "edge-corrected clarity", "high-definition full-field clarity"],
            "Viewing stability aid": ["freehand-only viewing", "textured steady-grip body", "image-stabilized viewing"],
        },
    },
    "Home Improvement & Hardware": {
        "Cordless drill": {
            "Fastening torque capability": ["light assembly torque", "general repair torque", "heavy fastening torque"],
            "Work-session battery endurance": ["short task endurance", "half-day project endurance", "full-day project endurance"],
        },
        "Claw hammer": {
            "Strike balance quality": ["front-heavy balance", "neutral workshop balance", "precision-tuned balance"],
            "Impact vibration damping": ["rigid impact transfer", "rubberized impact reduction", "multi-layer vibration isolation"],
        },
        "Adjustable wrench": {
            "Jaw adjustment precision": ["coarse jaw adjustment", "fine-thread jaw adjustment", "zero-play precision adjustment"],
            "Metal corrosion protection": ["unfinished steel surface", "protective chrome coating", "marine-grade protective coating"],
        },
        "Laser distance meter": {
            "Measurement distance span": ["single-room distance span", "whole-house distance span", "large-site distance span"],
            "Distance reading tolerance": ["centimeter-level tolerance", "five-millimeter tolerance", "millimeter-level tolerance"],
        },
        "Folding step ladder": {
            "Supported working load": ["light household load", "standard adult work load", "heavy equipment work load"],
            "Footing stabilization design": ["plain rubber feet", "wide anti-slip feet", "self-leveling stabilizer feet"],
        },
    },
    "Kitchen Appliances & Tools": {
        "Countertop air fryer": {
            "Cooking basket volume": ["single-serving basket", "small-family basket", "large-family basket"],
            "Hot-air circulation control": ["fixed fan circulation", "adaptive fan circulation", "dual-zone precision circulation"],
        },
        "Electric kettle": {
            "Water heating speed": ["standard heating cycle", "rapid heating cycle", "ultrafast heating cycle"],
            "Brew temperature selection": ["boil-only selection", "three preset temperatures", "degree-by-degree temperature control"],
        },
        "Hand blender": {
            "Blending motor strength": ["soft-food motor", "general blending motor", "heavy-duty crushing motor"],
            "Food-prep attachment range": ["blending shaft only", "shaft plus whisk", "multi-tool preparation set"],
        },
        "Digital kitchen scale": {
            "Ingredient weight resolution": ["whole-gram resolution", "half-gram resolution", "tenth-gram resolution"],
            "Maximum platform load": ["small-bowl load", "mixing-bowl load", "bulk-ingredient load"],
        },
        "Chef's knife": {
            "Blade edge retention": ["frequent-sharpening edge", "extended-use edge", "professional long-life edge"],
            "Cutting grip ergonomics": ["straight basic grip", "contoured balanced grip", "custom-fit anti-fatigue grip"],
        },
    },
    "Fitness & Sports Gear": {
        "Yoga exercise mat": {
            "Joint cushioning density": ["thin firm cushioning", "medium balanced cushioning", "thick impact-absorbing cushioning"],
            "Pose traction performance": ["basic dry traction", "sweat-resistant traction", "professional non-slip traction"],
        },
        "Resistance band set": {
            "Exercise tension range": ["light tension range", "light-to-heavy tension range", "rehab-to-athletic tension range"],
            "Band snap resistance": ["standard latex resistance", "reinforced latex resistance", "sleeved break-resistant construction"],
        },
        "Cast-iron kettlebell": {
            "Weight casting accuracy": ["approximate cast weight", "calibrated training weight", "competition-certified weight"],
            "Lift grip comfort": ["rough narrow handle", "smoothed wide handle", "ergonomic competition handle"],
        },
        "Speed jump rope": {
            "Handle rotation smoothness": ["friction rotation", "bushed rotation", "sealed-bearing rotation"],
            "Rope length adjustment": ["cut-to-length adjustment", "tool-adjusted length", "instant tool-free adjustment"],
        },
        "Muscle foam roller": {
            "Massage firmness selection": ["single soft firmness", "single medium firmness", "variable-zone firmness"],
            "Surface pressure contour": ["smooth surface", "shallow massage ridges", "multi-depth trigger contours"],
        },
    },
    "Pet Care & Animal Supplies": {
        "Dry cat food bag": {
            "Animal protein proportion": ["basic protein proportion", "high protein proportion", "meat-first protein proportion"],
            "Ingredient source disclosure": ["general ingredient listing", "named ingredient sources", "fully traceable ingredient sources"],
        },
        "Dog walking leash": {
            "Sustained pull tolerance": ["small-dog pull tolerance", "medium-dog pull tolerance", "large-dog pull tolerance"],
            "Collar clasp security": ["basic spring clasp", "locking swivel clasp", "double-secured climbing clasp"],
        },
        "Pet travel carrier": {
            "Airflow panel coverage": ["single ventilation panel", "three-sided ventilation", "full-surround ventilation"],
            "Carrier frame rigidity": ["soft unframed body", "reinforced flexible frame", "impact-resistant rigid frame"],
        },
        "Aquarium water filter": {
            "Tank flow adjustment": ["fixed water flow", "three-step water flow", "continuous precision flow"],
            "Water filtration stages": ["single mechanical stage", "mechanical plus carbon stages", "three-stage biological filtration"],
        },
        "Cat scratching tower": {
            "Scratching column height": ["kitten-height column", "adult-cat stretch column", "full-height climbing column"],
            "Tower base stability": ["compact lightweight base", "wide weighted base", "anti-tip anchored base"],
        },
    },
    "Gardening & Plant Care": {
        "Garden hand trowel": {
            "Digging blade rigidity": ["flexible stamped blade", "reinforced steel blade", "forged rigid blade"],
            "Soil release treatment": ["untreated blade surface", "polished release surface", "non-stick soil-shedding coating"],
        },
        "Pruning shears": {
            "Branch cutting diameter": ["thin-stem cutting", "medium-branch cutting", "thick-branch cutting"],
            "Handle return mechanism": ["manual handle reopening", "basic coil-spring return", "adjustable assisted return"],
        },
        "Garden watering can": {
            "Pour stream control": ["open-spout stream", "removable shower rose", "precision multi-pattern rose"],
            "Water reservoir volume": ["balcony-plant volume", "patio-garden volume", "large-bed garden volume"],
        },
        "Seed propagation tray": {
            "Reusable cell durability": ["single-season cells", "multi-season reinforced cells", "rigid nursery-grade cells"],
            "Root drainage layout": ["single drain opening", "multi-hole drainage", "air-pruning drainage channels"],
        },
        "Expandable garden hose": {
            "Hose kink resistance": ["basic flexible hose", "braided kink-resistant hose", "self-straightening anti-kink hose"],
            "Water pressure tolerance": ["low-pressure watering", "household mains pressure", "high-pressure outdoor use"],
        },
    },
    "Travel & Mobility Accessories": {
        "Hard-shell travel suitcase": {
            "Shell impact resistance": ["light-impact shell", "reinforced travel shell", "high-impact composite shell"],
            "Wheel movement quality": ["two fixed wheels", "four spinner wheels", "silent precision-bearing spinners"],
        },
        "Travel neck cushion": {
            "Neck support structure": ["soft wrap support", "contoured side support", "adjustable orthopedic support"],
            "Packed travel size": ["full-size carry", "compressible pouch size", "ultracompact roll size"],
        },
        "Packing organizer cube set": {
            "Clothing compression ability": ["organization only", "zippered light compression", "double-zip maximum compression"],
            "Cube seam durability": ["single-stitched seams", "reinforced double seams", "taped abrasion-resistant seams"],
        },
        "Portable luggage scale": {
            "Bag weighing capacity": ["cabin-bag capacity", "checked-bag capacity", "oversize-baggage capacity"],
            "Displayed weight accuracy": ["half-kilogram accuracy", "hundred-gram accuracy", "fifty-gram accuracy"],
        },
        "Passport document organizer": {
            "Travel document capacity": ["single-passport capacity", "couple travel capacity", "family document capacity"],
            "Personal data protection": ["standard fabric lining", "RFID-blocking lining", "shielded locking enclosure"],
        },
    },
    "Consumer Electronics": {
        "Portable Bluetooth speaker": {
            "Room audio projection": ["personal desk projection", "full-room projection", "outdoor gathering projection"],
            "Continuous playback duration": ["short-session playback", "all-day playback", "multi-day playback"],
        },
        "Wireless computer mouse": {
            "Cursor tracking precision": ["basic office tracking", "high-resolution tracking", "professional adjustable tracking"],
            "Click switch durability": ["standard click lifespan", "extended click lifespan", "rated esports click lifespan"],
        },
        "Portable power bank": {
            "Stored charging energy": ["single-phone recharge", "multiple-phone recharges", "laptop-capable energy reserve"],
            "Simultaneous charging outlets": ["one charging outlet", "two charging outlets", "multi-device fast-charge hub"],
        },
        "USB conference webcam": {
            "Video image definition": ["standard-definition video", "full-HD video", "ultra-HD video"],
            "Dim-room image handling": ["basic automatic exposure", "low-light enhancement", "sensor-level night correction"],
        },
        "Digital e-book reader": {
            "Reading display sharpness": ["entry-level text sharpness", "print-like text sharpness", "high-density premium sharpness"],
            "Offline library capacity": ["small personal library", "large personal library", "archive-scale library"],
        },
    },
    "Arts & Creative Materials": {
        "Watercolor paint palette": {
            "Paint pigment concentration": ["student pigment concentration", "artist pigment concentration", "professional pigment concentration"],
            "Color blending consistency": ["variable blending behavior", "reliable blending behavior", "studio-grade uniform blending"],
        },
        "Mixed-media sketchbook": {
            "Drawing paper weight": ["light sketch paper", "medium mixed-media paper", "heavy wet-media paper"],
            "Paper surface tooth": ["smooth pencil surface", "medium all-purpose tooth", "deep charcoal-friendly tooth"],
        },
        "Acrylic artist brush set": {
            "Bristle shape recovery": ["basic synthetic recovery", "resilient artist recovery", "precision shape-memory recovery"],
            "Brush profile variety": ["essential three profiles", "expanded studio profiles", "complete specialty profile range"],
        },
        "Reusable modeling clay kit": {
            "Clay color assortment": ["six-color assortment", "twelve-color assortment", "full-spectrum color assortment"],
            "Sculpted shape retention": ["soft temporary retention", "firm project retention", "detail-preserving long retention"],
        },
        "Embroidery beginner kit": {
            "Included thread variety": ["basic color bundle", "expanded color bundle", "shaded full-palette bundle"],
            "Stitching hoop stability": ["basic plastic hoop", "locking wooden hoop", "tension-controlled hoop frame"],
        },
    },
    "Learning & Educational Tools": {
        "Desktop world globe": {
            "Printed map detail": ["countries-only map detail", "cities-and-terrain detail", "reference-atlas map detail"],
            "Globe rotation mechanism": ["basic spindle rotation", "smooth axis rotation", "dual-axis precision rotation"],
        },
        "Home science experiment kit": {
            "Experiment topic breadth": ["single-topic experiments", "multi-topic laboratory set", "cross-discipline project library"],
            "Procedure explanation clarity": ["brief instruction cards", "illustrated step guidance", "concept-rich guided curriculum"],
        },
        "Magnetic construction tile set": {
            "Tile connection strength": ["light magnetic connection", "reinforced magnetic connection", "load-bearing magnetic connection"],
            "Construction shape assortment": ["basic square-triangle set", "expanded geometric set", "advanced architectural shape set"],
        },
        "Language vocabulary card set": {
            "Vocabulary topic coverage": ["survival vocabulary", "daily conversation vocabulary", "comprehensive thematic vocabulary"],
            "Review sequencing system": ["unordered card stack", "level-grouped review", "spaced-repetition review indexing"],
        },
        "Beginner optical microscope": {
            "Specimen enlargement range": ["low classroom enlargement", "standard biology enlargement", "high-detail cellular enlargement"],
            "Image focus adjustment": ["single coarse focus", "coarse and fine focus", "precision dual-speed focus"],
        },
    },
}


def flatten_goods(goods: dict) -> list[tuple[str, str, dict]]:
    return [(category, name, attrs) for category, items in goods.items() for name, attrs in items.items()]


def validate_novelty() -> None:
    original = json.loads((ROOT / "everyday_goods_full.json").read_text(encoding="utf-8"))
    old_categories = set(original)
    old_goods = {name for items in original.values() for name in items}
    old_attrs = {attr for items in original.values() for attrs in items.values() for attr in attrs}
    old_values = {
        str(value)
        for items in original.values()
        for attrs in items.values()
        for values in attrs.values()
        for value in values
    }

    rows = flatten_goods(GOODS)
    new_categories = set(GOODS)
    new_goods = {name for _, name, _ in rows}
    new_attrs = {attr for _, _, attrs in rows for attr in attrs}
    new_values = {str(value) for _, _, attrs in rows for values in attrs.values() for value in values}

    assert len(rows) == 50, f"Expected 50 goods, found {len(rows)}"
    assert len(new_categories) == 10
    assert len(new_goods) == 50
    assert len(new_attrs) == 100
    assert not (new_categories & old_categories), "New category overlaps original category"
    assert not (new_goods & old_goods), "New good overlaps original good"
    assert not (new_attrs & old_attrs), "New attribute name overlaps original attribute name"
    assert not (new_values & old_values), "New attribute value overlaps original attribute value"

    for _, name, attrs in rows:
        assert len(attrs) == 2, f"{name} must have exactly two attributes"
        for attr, values in attrs.items():
            assert len(values) == 3, f"{name}/{attr} must have exactly three levels"
            assert len(set(values)) == 3, f"{name}/{attr} contains duplicate levels"


def build_cases() -> list[list]:
    """Use every unordered pair and a deterministic, globally balanced code schedule."""
    n_goods = len(flatten_goods(GOODS))
    codes = list(range(81))
    random.Random(SEED).shuffle(codes)

    cases = []
    pair_index = 0
    for x_num in range(n_goods):
        for y_num in range(x_num + 1, n_goods):
            start = (pair_index * CONFIGS_PER_PAIR) % len(codes)
            selected = [codes[(start + offset) % len(codes)] for offset in range(CONFIGS_PER_PAIR)]
            assert len(set(selected)) == CONFIGS_PER_PAIR
            cases.append([x_num, y_num, selected])
            pair_index += 1
    return cases


def build_markdown(cases: list[list]) -> str:
    rows = flatten_goods(GOODS)
    lines = [
        "# OOD New-Goods Evaluation Suite (50 Goods)",
        "",
        "> Evaluation-only: do not use this suite for training, reward construction, hyperparameter tuning, or checkpoint selection.",
        "",
        "## Design summary",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| New categories | {len(GOODS)} |",
        f"| New goods | {len(rows)} |",
        f"| New attribute names | {sum(len(attrs) for _, _, attrs in rows)} |",
        f"| Unordered goods pairs | {len(cases):,} |",
        f"| Configurations per pair | {CONFIGS_PER_PAIR} |",
        f"| Structural cases | {sum(len(entry[2]) for entry in cases):,} |",
        f"| X/Y prompts | {2 * sum(len(entry[2]) for entry in cases):,} |",
        f"| Deterministic generation seed | {SEED} |",
        "",
        "All category names, good names, attribute names, and complete attribute-value strings were checked for exact overlap against `everyday_goods_full.json`; all overlap counts are zero.",
        "",
        "## Goods and attributes",
        "",
        "| # | Category | Good | Attribute 1 (low → high) | Attribute 2 (low → high) |",
        "|---:|---|---|---|---|",
    ]
    for idx, (category, name, attrs) in enumerate(rows, start=1):
        (attr1, vals1), (attr2, vals2) = attrs.items()
        a1 = f"**{attr1}:** " + " → ".join(vals1)
        a2 = f"**{attr2}:** " + " → ".join(vals2)
        lines.append(f"| {idx} | {category} | {name} | {a1} | {a2} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    validate_novelty()
    cases = build_cases()

    goods_path = DATA_DIR / "ood_new_goods_50.json"
    cases_path = DATA_DIR / "ood_new_goods_50_test.json"
    table_path = DATA_DIR / "OOD_NEW_GOODS_50_TABLE.md"

    goods_path.write_text(json.dumps(GOODS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cases_path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    table_path.write_text(build_markdown(cases), encoding="utf-8")

    print(f"Wrote {goods_path}: 50 goods")
    print(f"Wrote {cases_path}: {len(cases):,} pairs, {sum(len(x[2]) for x in cases):,} cases")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
