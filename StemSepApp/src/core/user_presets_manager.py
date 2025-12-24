"""
User Presets Manager: Save, load, and manage custom user presets.

This module allows users to create, edit, and manage custom presets beyond
the 15 built-in presets, enabling personalized workflows and configurations.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class UserPreset:
    """User-defined preset configuration."""
    id: str
    name: str
    description: str
    model_ids: List[str]  # List of model IDs to use
    workflow: str  # 'single', 'ensemble', 'chain'
    settings: Dict[str, Any]  # Model-specific settings
    use_case: List[str]  # Use case tags
    created_date: str
    modified_date: str
    is_custom: bool = True
    vram_required: Optional[float] = None
    speed: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreset':
        """Create instance from dictionary."""
        return cls(**data)


class UserPresetsManager:
    """
    Manages user-defined presets for stem separation.
    
    Features:
    - Create custom presets with model selections and settings
    - Save/load/edit/delete user presets
    - Export presets to files for sharing
    - Import presets from files
    - Validate preset configurations
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize UserPresetsManager.
        
        Args:
            config_dir: Directory for storing preset files (default: ~/.stemsep)
        """
        self.config_dir = config_dir or (Path.home() / ".stemsep")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Preset files
        self.presets_file = self.config_dir / "user_presets.json"
        
        # Load existing presets
        self.presets: Dict[str, UserPreset] = {}
        self._load_presets()
        
        logger.info(f"UserPresetsManager initialized. Loaded {len(self.presets)} presets")
    
    def create_preset(
        self,
        name: str,
        description: str,
        model_ids: List[str],
        workflow: str = "single",
        settings: Optional[Dict[str, Any]] = None,
        use_case: Optional[List[str]] = None,
        vram_required: Optional[float] = None,
        speed: Optional[str] = None
    ) -> str:
        """
        Create a new user preset.
        
        Args:
            name: Preset name
            description: Preset description
            model_ids: List of model IDs to use
            workflow: Workflow type ('single', 'ensemble', 'chain')
            settings: Model-specific settings
            use_case: Use case tags
            vram_required: Estimated VRAM requirement (GB)
            speed: Estimated processing speed
            
        Returns:
            Preset ID
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not name or name.strip() == '':
            raise ValueError("Preset name cannot be empty")
        
        if not description or description.strip() == '':
            raise ValueError("Preset description cannot be empty")
        
        if not model_ids or len(model_ids) == 0:
            raise ValueError("Model IDs list cannot be empty")
        
        if workflow not in ['single', 'ensemble', 'chain']:
            raise ValueError(f"Workflow must be 'single', 'ensemble', or 'chain', got '{workflow}'")
        
        # Generate preset ID
        preset_id = self._generate_preset_id(name)
        
        # Create preset
        now = datetime.now().isoformat()
        preset = UserPreset(
            id=preset_id,
            name=name,
            description=description,
            model_ids=model_ids,
            workflow=workflow,
            settings=settings or {},
            use_case=use_case or [],
            created_date=now,
            modified_date=now,
            vram_required=vram_required,
            speed=speed
        )
        
        # Save preset
        self.presets[preset_id] = preset
        self._save_presets()
        
        logger.info(f"Created preset '{name}' with ID: {preset_id}")
        return preset_id
    
    def get_preset(self, preset_id: str) -> Optional[UserPreset]:
        """
        Get a preset by ID.
        
        Args:
            preset_id: Preset ID
            
        Returns:
            UserPreset if found, None otherwise
            
        Raises:
            ValueError: If preset_id is empty
        """
        if not preset_id or preset_id.strip() == '':
            raise ValueError("Preset ID cannot be empty")
        
        return self.presets.get(preset_id)
    
    def list_presets(self) -> List[UserPreset]:
        """
        List all user presets.
        
        Returns:
            List of UserPreset objects, sorted by name
        """
        return sorted(self.presets.values(), key=lambda p: p.name)
    
    def update_preset(
        self,
        preset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
        workflow: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        use_case: Optional[List[str]] = None,
        vram_required: Optional[float] = None,
        speed: Optional[str] = None
    ) -> bool:
        """
        Update an existing preset.
        
        Args:
            preset_id: ID of preset to update
            name: New name (optional)
            description: New description (optional)
            model_ids: New model IDs (optional)
            workflow: New workflow (optional)
            settings: New settings (optional)
            use_case: New use cases (optional)
            vram_required: New VRAM requirement (optional)
            speed: New speed estimate (optional)
            
        Returns:
            True if updated successfully, False if preset not found
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not preset_id or preset_id.strip() == '':
            raise ValueError("Preset ID cannot be empty")
        
        if preset_id not in self.presets:
            return False
        
        preset = self.presets[preset_id]
        
        # Update fields
        if name is not None:
            if name.strip() == '':
                raise ValueError("Preset name cannot be empty")
            preset.name = name
        
        if description is not None:
            if description.strip() == '':
                raise ValueError("Preset description cannot be empty")
            preset.description = description
        
        if model_ids is not None:
            if len(model_ids) == 0:
                raise ValueError("Model IDs list cannot be empty")
            preset.model_ids = model_ids
        
        if workflow is not None:
            if workflow not in ['single', 'ensemble', 'chain']:
                raise ValueError(f"Workflow must be 'single', 'ensemble', or 'chain', got '{workflow}'")
            preset.workflow = workflow
        
        if settings is not None:
            preset.settings = settings
        
        if use_case is not None:
            preset.use_case = use_case
        
        if vram_required is not None:
            preset.vram_required = vram_required
        
        if speed is not None:
            preset.speed = speed
        
        # Update modification date
        preset.modified_date = datetime.now().isoformat()
        
        # Save changes
        self._save_presets()
        
        logger.info(f"Updated preset: {preset_id}")
        return True
    
    def delete_preset(self, preset_id: str) -> bool:
        """
        Delete a preset.
        
        Args:
            preset_id: ID of preset to delete
            
        Returns:
            True if deleted successfully, False if preset not found
            
        Raises:
            ValueError: If preset_id is empty
        """
        if not preset_id or preset_id.strip() == '':
            raise ValueError("Preset ID cannot be empty")
        
        if preset_id not in self.presets:
            return False
        
        del self.presets[preset_id]
        self._save_presets()
        
        logger.info(f"Deleted preset: {preset_id}")
        return True
    
    def export_preset(self, preset_id: str, export_path: Path) -> None:
        """
        Export a preset to a file.
        
        Args:
            preset_id: ID of preset to export
            export_path: Path to export file
            
        Raises:
            ValueError: If preset not found or path is invalid
        """
        if not preset_id or preset_id.strip() == '':
            raise ValueError("Preset ID cannot be empty")
        
        if preset_id not in self.presets:
            raise ValueError(f"Preset not found: {preset_id}")
        
        export_path = Path(export_path)
        
        # Ensure directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export preset
        preset = self.presets[preset_id]
        export_data = {
            'version': 1,
            'preset': preset.to_dict()
        }
        
        try:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported preset '{preset.name}' to {export_path}")
        except Exception as e:
            raise ValueError(f"Failed to export preset: {str(e)}")
    
    def import_preset(self, import_path: Path) -> str:
        """
        Import a preset from a file.
        
        Args:
            import_path: Path to import file
            
        Returns:
            Preset ID of imported preset
            
        Raises:
            ValueError: If import fails or file is invalid
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise ValueError(f"Import file not found: {import_path}")
        
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if 'preset' not in import_data:
                raise ValueError("Invalid preset file: missing 'preset' key")
            
            # Create preset from imported data
            preset_data = import_data['preset']
            
            # Generate new ID to avoid conflicts
            name = preset_data.get('name', 'Imported Preset')
            preset_id = self._generate_preset_id(name)
            
            # Update dates
            now = datetime.now().isoformat()
            preset_data['id'] = preset_id
            preset_data['created_date'] = now
            preset_data['modified_date'] = now
            preset_data['is_custom'] = True
            
            # Create preset
            preset = UserPreset.from_dict(preset_data)
            self.presets[preset_id] = preset
            self._save_presets()
            
            logger.info(f"Imported preset '{preset.name}' with ID: {preset_id}")
            return preset_id
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in preset file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to import preset: {str(e)}")
    
    def export_all_presets(self, export_path: Path) -> None:
        """
        Export all presets to a single file.
        
        Args:
            export_path: Path to export file
            
        Raises:
            ValueError: If export fails
        """
        export_path = Path(export_path)
        
        # Ensure directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export all presets
        export_data = {
            'version': 1,
            'presets': {pid: preset.to_dict() for pid, preset in self.presets.items()}
        }
        
        try:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(self.presets)} presets to {export_path}")
        except Exception as e:
            raise ValueError(f"Failed to export presets: {str(e)}")
    
    def import_all_presets(self, import_path: Path, merge: bool = False) -> int:
        """
        Import multiple presets from a file.
        
        Args:
            import_path: Path to import file
            merge: If True, merge with existing presets; if False, replace all
            
        Returns:
            Number of presets imported
            
        Raises:
            ValueError: If import fails
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise ValueError(f"Import file not found: {import_path}")
        
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if 'presets' not in import_data:
                raise ValueError("Invalid presets file: missing 'presets' key")
            
            # Clear existing presets if not merging
            if not merge:
                self.presets.clear()
            
            # Import presets
            imported_count = 0
            for preset_data in import_data['presets'].values():
                # Generate new ID if conflicts exist
                preset_id = preset_data.get('id', '')
                if preset_id in self.presets:
                    preset_id = self._generate_preset_id(preset_data.get('name', 'Preset'))
                    preset_data['id'] = preset_id
                
                preset = UserPreset.from_dict(preset_data)
                self.presets[preset_id] = preset
                imported_count += 1
            
            self._save_presets()
            
            logger.info(f"Imported {imported_count} presets from {import_path}")
            return imported_count
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in presets file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to import presets: {str(e)}")
    
    def validate_preset(self, preset_id: str) -> Dict[str, Any]:
        """
        Validate a preset configuration.
        
        Args:
            preset_id: ID of preset to validate
            
        Returns:
            Dictionary with validation results: {'valid': bool, 'errors': list, 'warnings': list}
            
        Raises:
            ValueError: If preset_id is empty or preset not found
        """
        if not preset_id or preset_id.strip() == '':
            raise ValueError("Preset ID cannot be empty")
        
        if preset_id not in self.presets:
            raise ValueError(f"Preset not found: {preset_id}")
        
        preset = self.presets[preset_id]
        errors = []
        warnings = []
        
        # Check required fields
        if not preset.name:
            errors.append("Preset name is missing")
        
        if not preset.model_ids:
            errors.append("No models specified")
        
        if not preset.workflow:
            errors.append("Workflow type is missing")
        
        # Check workflow consistency
        if preset.workflow == 'single' and len(preset.model_ids) > 1:
            warnings.append("Single workflow should use only one model")
        
        if preset.workflow == 'ensemble' and len(preset.model_ids) < 2:
            warnings.append("Ensemble workflow typically uses multiple models")
        
        # Check VRAM
        if preset.vram_required and preset.vram_required < 0:
            errors.append("VRAM requirement cannot be negative")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    # ---------- Private Methods ----------
    
    def _load_presets(self) -> None:
        """Load presets from JSON file."""
        if not self.presets_file.exists():
            logger.info("No user presets file found, starting fresh")
            return
        
        try:
            with open(self.presets_file, 'r') as f:
                data = json.load(f)
            
            for preset_data in data.get('presets', {}).values():
                preset = UserPreset.from_dict(preset_data)
                self.presets[preset.id] = preset
            
            logger.info(f"Loaded {len(self.presets)} user presets")
        except Exception as e:
            logger.error(f"Failed to load user presets: {str(e)}")
    
    def _save_presets(self) -> None:
        """Save presets to JSON file."""
        try:
            data = {
                'version': 1,
                'presets': {pid: preset.to_dict() for pid, preset in self.presets.items()}
            }
            
            with open(self.presets_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.presets)} user presets")
        except Exception as e:
            logger.error(f"Failed to save user presets: {str(e)}")
            raise
    
    def _generate_preset_id(self, name: str) -> str:
        """
        Generate a unique preset ID.
        
        Args:
            name: Preset name
            
        Returns:
            Unique preset ID
        """
        # Create base ID from name
        base_id = "user_" + name.lower().replace(' ', '_')
        
        # Remove special characters
        base_id = ''.join(c if c.isalnum() or c == '_' else '_' for c in base_id)
        
        # Ensure uniqueness
        preset_id = base_id
        counter = 1
        while preset_id in self.presets:
            preset_id = f"{base_id}_{counter}"
            counter += 1
        
        return preset_id
