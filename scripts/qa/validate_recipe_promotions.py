import json
import sys
from pathlib import Path


REQUIRED_THRESHOLD_KEYS = (
    "min_correlation",
    "min_snr_db",
    "min_si_sdr_db",
    "max_gain_delta_db",
    "max_clipped_samples",
)


def load_recipes(repo_root: Path) -> list[dict]:
    path = repo_root / "StemSepApp" / "assets" / "recipes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    recipes = data.get("recipes")
    if not isinstance(recipes, list):
        raise SystemExit(f"recipes.json did not contain a recipes array: {path}")
    return recipes


def validate_recipe(recipe: dict) -> list[str]:
    errors: list[str] = []
    recipe_id = str(recipe.get("id") or "<unknown>")
    requires_qa_pass = recipe.get("requires_qa_pass") is True
    if not requires_qa_pass:
        return errors

    golden_set_id = recipe.get("golden_set_id")
    if not isinstance(golden_set_id, str) or not golden_set_id.strip():
        errors.append(f"{recipe_id}: missing golden_set_id")

    thresholds = recipe.get("audio_quality_thresholds")
    if not isinstance(thresholds, dict):
        errors.append(f"{recipe_id}: missing audio_quality_thresholds object")
        return errors

    for key in REQUIRED_THRESHOLD_KEYS:
        if key not in thresholds:
            errors.append(f"{recipe_id}: audio_quality_thresholds missing {key}")
            continue
        value = thresholds[key]
        if key == "max_clipped_samples":
            if not isinstance(value, int) or value < 0:
                errors.append(
                    f"{recipe_id}: audio_quality_thresholds.{key} must be a non-negative integer"
                )
        else:
            if not isinstance(value, (int, float)):
                errors.append(
                    f"{recipe_id}: audio_quality_thresholds.{key} must be numeric"
                )

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    recipes = load_recipes(repo_root)

    failures: list[str] = []
    enforced = 0
    for recipe in recipes:
        if recipe.get("requires_qa_pass") is True:
            enforced += 1
        failures.extend(validate_recipe(recipe))

    if failures:
        print("Recipe promotion policy validation failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(
        f"Recipe promotion policies valid. Checked {enforced} recipes that require QA promotion."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
