import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

class RecipeManager:
    """Manages separation recipes."""
    
    def __init__(self, assets_dir: Path):
        self.logger = logging.getLogger(__name__)
        self.assets_dir = assets_dir
        self.recipes: List[Dict] = []
        self.load_recipes()

    def load_recipes(self):
        """Load recipes from JSON file."""
        recipe_path = self.assets_dir / "recipes.json"
        if not recipe_path.exists():
            self.logger.warning(f"Recipes file not found at {recipe_path}")
            return

        try:
            with open(recipe_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.recipes = data.get("recipes", [])
            self.logger.info(f"Loaded {len(self.recipes)} recipes")
        except Exception as e:
            self.logger.error(f"Failed to load recipes: {e}")

    def get_recipe(self, recipe_id: str) -> Optional[Dict]:
        """Get recipe by ID."""
        return next((r for r in self.recipes if r["id"] == recipe_id), None)

    def get_all_recipes(self) -> List[Dict]:
        return self.recipes

    def recipe_to_ensemble_config(self, recipe_id: str) -> Optional[Dict]:
        """Convert a recipe to an ensemble configuration for SeparationManager."""
        recipe = self.get_recipe(recipe_id)
        if not recipe:
            return None
            
        # Convert steps to ensemble config format
        # SeparationManager expects: [{'model_id': '...', 'weight': ...}, ...]
        # Recipe has: [{'model_id': ..., 'weight': ..., 'role': ...}]
        
        ensemble_config = []
        for step in recipe['steps']:
            ensemble_config.append({
                'model_id': step['model_id'],
                'weight': step.get('weight', 1.0)
            })
            
        return {
            'ensemble_config': ensemble_config,
            'algorithm': recipe.get('algorithm', 'average'),
            'defaults': recipe.get('defaults', {})
        }
