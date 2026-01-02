"""
Output path management for separated audio files.

This module provides functionality to manage output directories, validate paths,
create directories, apply naming templates, and persist user preferences.
"""

import logging
import json
import os
import psutil
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputPathManager:
    """
    Manages output paths for separated audio files.
    
    Handles path validation, creation, templates, persistence of preferences,
    and disk space checking.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize OutputPathManager.
        
        Args:
            config_dir: Directory to store configuration. Defaults to ~/.stemsep
        """
        if config_dir is None:
            config_dir = Path.home() / ".stemsep"
        
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "output_config.json"
        self._config = self._load_config()
        
        logger.debug(f"OutputPathManager initialized with config dir: {self.config_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            dict: Configuration dictionary.
        """
        if not self.config_file.exists():
            logger.debug("No config file found, using defaults")
            return {
                "last_output_path": str(Path.home() / "Music" / "StemSep"),
                "default_template": "{original_name}_{stem}",
                "create_subdirs": True,
                "min_free_space_gb": 1.0
            }
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logger.debug(f"Loaded config from {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _save_config(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            logger.debug(f"Saved config to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def validate_path(self, path: str) -> Dict[str, Any]:
        """
        Validate an output path.
        
        Args:
            path: Path to validate.
            
        Returns:
            dict: Validation result containing:
                - valid (bool): Whether path is valid
                - exists (bool): Whether path exists
                - writable (bool): Whether path is writable
                - free_space_gb (float): Free space in GB
                - reason (str): Explanation if not valid
                
        Raises:
            ValueError: If path is empty or None.
            
        Example:
            >>> manager = OutputPathManager()
            >>> result = manager.validate_path("/home/user/music")
            >>> if result['valid']:
            ...     print(f"{result['free_space_gb']} GB available")
        """
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string")
        
        logger.debug(f"Validating path: {path}")
        
        result = {
            "valid": False,
            "exists": False,
            "writable": False,
            "free_space_gb": 0.0,
            "reason": ""
        }
        
        try:
            path_obj = Path(path).expanduser().resolve()
            
            # Check if path exists
            result["exists"] = path_obj.exists()
            
            # Check parent directory if path doesn't exist
            check_path = path_obj if result["exists"] else path_obj.parent
            
            # Check if we can write to this location
            if check_path.exists():
                result["writable"] = os.access(check_path, os.W_OK)
            else:
                # Check if we can create it
                try:
                    check_path.mkdir(parents=True, exist_ok=True)
                    result["writable"] = True
                    result["exists"] = True
                    logger.debug(f"Created directory: {check_path}")
                except Exception as e:
                    result["reason"] = f"Cannot create directory: {e}"
                    logger.warning(result["reason"])
                    return result
            
            # Get free space
            try:
                usage = psutil.disk_usage(str(check_path))
                result["free_space_gb"] = round(usage.free / (1024**3), 2)
            except Exception as e:
                logger.warning(f"Could not get disk space: {e}")
                result["free_space_gb"] = 0.0
            
            # Check minimum free space
            min_space = self._config.get("min_free_space_gb", 1.0)
            if result["free_space_gb"] < min_space:
                result["reason"] = f"Insufficient space. Need {min_space} GB, have {result['free_space_gb']} GB"
                logger.warning(result["reason"])
                return result
            
            # All checks passed
            result["valid"] = True
            result["reason"] = "Path is valid"
            logger.info(f"Path validated successfully: {path}")
            
        except Exception as e:
            result["reason"] = f"Invalid path: {e}"
            logger.error(result["reason"])
        
        return result
    
    def set_output_path(self, path: str, validate: bool = True) -> bool:
        """
        Set the output path and save to config.
        
        Args:
            path: Output path to set.
            validate: Whether to validate path before setting.
            
        Returns:
            bool: True if successful, False otherwise.
            
        Raises:
            ValueError: If path is empty or validation fails.
            
        Example:
            >>> manager = OutputPathManager()
            >>> manager.set_output_path("/home/user/music/separated")
        """
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string")
        
        if validate:
            validation = self.validate_path(path)
            if not validation["valid"]:
                raise ValueError(f"Invalid path: {validation['reason']}")
        
        self._config["last_output_path"] = str(Path(path).expanduser().resolve())
        success = self._save_config()
        
        if success:
            logger.info(f"Output path set to: {self._config['last_output_path']}")
        
        return success
    
    def get_output_path(self) -> str:
        """
        Get the current output path.
        
        Returns:
            str: Current output path.
            
        Example:
            >>> manager = OutputPathManager()
            >>> path = manager.get_output_path()
        """
        return self._config.get("last_output_path", str(Path.home() / "Music" / "StemSep"))
    
    def create_output_directory(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create output directory structure.
        
        Args:
            path: Path to create. Uses saved path if None.
            
        Returns:
            dict: Result containing:
                - success (bool): Whether creation succeeded
                - path (str): Created path
                - message (str): Success or error message
                
        Example:
            >>> manager = OutputPathManager()
            >>> result = manager.create_output_directory()
            >>> if result['success']:
            ...     print(f"Created: {result['path']}")
        """
        if path is None:
            path = self.get_output_path()
        
        logger.debug(f"Creating output directory: {path}")
        
        try:
            path_obj = Path(path).expanduser().resolve()
            path_obj.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created output directory: {path_obj}")
            return {
                "success": True,
                "path": str(path_obj),
                "message": "Directory created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return {
                "success": False,
                "path": path,
                "message": f"Error: {e}"
            }
    
    def apply_naming_template(
        self,
        template: str,
        original_filename: str,
        stem_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Apply naming template to generate output filename.
        
        Args:
            template: Template string with placeholders:
                - {original_name}: Original filename without extension
                - {stem}: Stem name (vocals, instrumental, etc)
                - {timestamp}: Current timestamp
                - {date}: Current date (YYYY-MM-DD)
                - {artist}, {album}, {title}: From metadata if provided
            original_filename: Original audio filename.
            stem_name: Name of the stem being saved.
            metadata: Optional metadata dictionary.
            
        Returns:
            str: Formatted filename.
            
        Raises:
            ValueError: If required parameters are empty.
            
        Example:
            >>> manager = OutputPathManager()
            >>> name = manager.apply_naming_template(
            ...     "{original_name}_{stem}",
            ...     "song.mp3",
            ...     "vocals"
            ... )
            >>> print(name)  # "song_vocals.wav"
        """
        if not template or not isinstance(template, str):
            raise ValueError("Template must be a non-empty string")
        if not original_filename or not isinstance(original_filename, str):
            raise ValueError("Original filename must be a non-empty string")
        if not stem_name or not isinstance(stem_name, str):
            raise ValueError("Stem name must be a non-empty string")
        
        logger.debug(f"Applying template: {template}")
        
        # Extract original name without extension
        original_name = Path(original_filename).stem
        
        # Build replacement dictionary
        replacements = {
            "original_name": original_name,
            "stem": stem_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Add metadata if provided
        if metadata:
            replacements.update({
                "artist": metadata.get("artist", "Unknown"),
                "album": metadata.get("album", "Unknown"),
                "title": metadata.get("title", original_name)
            })
        
        # Apply replacements
        result = template
        for key, value in replacements.items():
            result = result.replace(f"{{{key}}}", value)
        
        # Add .wav extension if not present
        if not result.endswith(('.wav', '.flac', '.mp3')):
            result += ".wav"
        
        logger.debug(f"Template result: {result}")
        return result
    
    def generate_full_path(
        self,
        original_filename: str,
        stem_name: str,
        output_dir: Optional[str] = None,
        template: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        create_subdir: Optional[bool] = None
    ) -> str:
        """
        Generate full output path for a stem file.
        
        Args:
            original_filename: Original audio filename.
            stem_name: Name of the stem.
            output_dir: Output directory. Uses saved path if None.
            template: Naming template. Uses default if None.
            metadata: Optional metadata for template.
            create_subdir: Whether to create subdirectory per song. Uses config if None.
            
        Returns:
            str: Full path for output file.
            
        Example:
            >>> manager = OutputPathManager()
            >>> path = manager.generate_full_path("song.mp3", "vocals")
            >>> print(path)  # "/home/user/Music/StemSep/song/song_vocals.wav"
        """
        # Get output directory
        if output_dir is None:
            output_dir = self.get_output_path()
        
        # Get template
        if template is None:
            template = self._config.get("default_template", "{original_name}_{stem}")
        
        # Check if we should create subdirectory
        if create_subdir is None:
            create_subdir = self._config.get("create_subdirs", True)
        
        # Build path
        base_path = Path(output_dir).expanduser().resolve()
        
        # Add subdirectory for song if enabled
        if create_subdir:
            song_name = Path(original_filename).stem
            base_path = base_path / song_name
        
        # Generate filename
        filename = self.apply_naming_template(template, original_filename, stem_name, metadata)
        
        # Combine
        full_path = base_path / filename
        
        logger.debug(f"Generated full path: {full_path}")
        return str(full_path)
    
    def set_naming_template(self, template: str) -> bool:
        """
        Set the default naming template.
        
        Args:
            template: Template string.
            
        Returns:
            bool: True if successful.
            
        Raises:
            ValueError: If template is empty.
        """
        if not template or not isinstance(template, str):
            raise ValueError("Template must be a non-empty string")
        
        self._config["default_template"] = template
        success = self._save_config()
        
        if success:
            logger.info(f"Naming template set to: {template}")
        
        return success
    
    def get_naming_template(self) -> str:
        """
        Get the current naming template.
        
        Returns:
            str: Current template.
        """
        return self._config.get("default_template", "{original_name}_{stem}")
    
    def set_min_free_space(self, gb: float) -> bool:
        """
        Set minimum required free space in GB.
        
        Args:
            gb: Minimum free space in GB.
            
        Returns:
            bool: True if successful.
            
        Raises:
            ValueError: If gb is negative.
        """
        if gb < 0:
            raise ValueError("Minimum free space must be non-negative")
        
        self._config["min_free_space_gb"] = gb
        success = self._save_config()
        
        if success:
            logger.info(f"Minimum free space set to: {gb} GB")
        
        return success
    
    def check_space_for_files(
        self,
        audio_file_sizes: list,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if there's enough space for output files.
        
        Args:
            audio_file_sizes: List of input file sizes in bytes.
            output_path: Path to check. Uses saved path if None.
            
        Returns:
            dict: Result containing:
                - sufficient (bool): Whether there's enough space
                - required_gb (float): Required space in GB
                - available_gb (float): Available space in GB
                - message (str): Explanation
                
        Example:
            >>> manager = OutputPathManager()
            >>> result = manager.check_space_for_files([100_000_000])  # 100MB
            >>> if result['sufficient']:
            ...     print("Enough space!")
        """
        if output_path is None:
            output_path = self.get_output_path()
        
        # Estimate output size (stems are typically similar size to input)
        total_input_bytes = sum(audio_file_sizes)
        # Assume 4 stems per file on average
        estimated_output_bytes = total_input_bytes * 4
        required_gb = round(estimated_output_bytes / (1024**3), 2)
        
        # Get available space
        try:
            path_obj = Path(output_path).expanduser().resolve()
            if not path_obj.exists():
                path_obj = path_obj.parent
            
            usage = psutil.disk_usage(str(path_obj))
            available_gb = round(usage.free / (1024**3), 2)
            
            # Add buffer from config
            min_space = self._config.get("min_free_space_gb", 1.0)
            required_with_buffer = required_gb + min_space
            
            sufficient = available_gb >= required_with_buffer
            
            if sufficient:
                message = f"Sufficient space available ({available_gb} GB available, {required_gb} GB needed)"
            else:
                message = f"Insufficient space ({available_gb} GB available, {required_with_buffer} GB needed including buffer)"
            
            logger.info(message)
            
            return {
                "sufficient": sufficient,
                "required_gb": required_gb,
                "available_gb": available_gb,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return {
                "sufficient": False,
                "required_gb": required_gb,
                "available_gb": 0.0,
                "message": f"Error checking space: {e}"
            }


if __name__ == "__main__":
    # Quick test/demo
    logging.basicConfig(level=logging.INFO)
    
    manager = OutputPathManager()
    
    print("\n=== Output Path Manager Demo ===\n")
    
    # Get current path
    current_path = manager.get_output_path()
    print(f"Current output path: {current_path}")
    
    # Validate path
    validation = manager.validate_path(current_path)
    print(f"\nPath validation:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Writable: {validation['writable']}")
    print(f"  Free space: {validation['free_space_gb']} GB")
    
    # Test naming template
    filename = manager.apply_naming_template(
        "{original_name}_{stem}",
        "my_song.mp3",
        "vocals"
    )
    print(f"\nGenerated filename: {filename}")
    
    # Generate full path
    full_path = manager.generate_full_path("my_song.mp3", "vocals")
    print(f"Full output path: {full_path}")
