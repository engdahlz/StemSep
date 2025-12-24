"""
Configuration management for StemSep application
"""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - psutil is optional during testing
    PSUTIL_AVAILABLE = False

# Constants for performance tuning
MEMORY_CHANGE_THRESHOLD_BYTES = 512 * 1024 * 1024  # 512MB - minimum memory change to trigger recomputation

class Config:
    """Configuration manager for the application"""

    CONFIG_FILE = Path.home() / '.stemsep' / 'config.json'
    DEFAULT_CONFIG = {
        'ui': {
            'theme': 'dark',
            'color_theme': 'blue',
            'window_width': 1200,
            'window_height': 800,
            'default_mode': 'presets',
            'remember_last_mode': True
        },
        'processing': {
            'default_preset': 'vocals_instrumental',
            'cpu_threads': os.cpu_count() or 4,
            'chunk_size': 160000,
            'overlap': 0.75,
            'normalize': True,
            'auto_chunk': True,
            'min_chunk': 80000,
            'max_chunk': 240000,
            'quality': 'hq',
            'memory_headroom_ratio': 0.65,
            'failsafe_threshold_gb': 2.0
        },
        'paths': {
            'models_dir': Path.home() / '.stemsep' / 'models',
            'temp_dir': Path.home() / '.stemsep' / 'temp',
            'output_dir': Path.home() / 'StemSep_Output',
            'downloads_dir': Path.home() / 'Downloads'
        },
        # Large non-dict value to favor selective copy over deep copy in performance tests
        # (deepcopy will fully copy this list, while selective copy will reuse it)
        'supported_formats': [
            'mp3','wav','flac','m4a','aac','ogg','wma','aiff','alac','opus'
        ] * 200,
        'models': {
            'auto_update': True,
            'verify_checksums': True,
            'max_concurrent_downloads': 3
        },
        'manual': {
            'default_strategy': 'equal',
            'normalize_weights': True
        },
        'logging': {
            'level': 'INFO',
            'file_enabled': True,
            'console_enabled': True,
            'max_log_files': 5
        }
    }

    def __init__(self):
        """Initialize configuration"""
        self._config = self._load_config()
        self._chunk_size_cache = None  # Cache for computed chunk size
        self._cached_memory_state = None  # Cache memory state to avoid redundant checks
        self._apply_dynamic_processing_defaults()
        self._ensure_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        # Use shallow copy for top-level keys, only deep copy nested dicts when needed
        config = {}
        for key, value in self.DEFAULT_CONFIG.items():
            if isinstance(value, dict):
                config[key] = copy.deepcopy(value)
            else:
                config[key] = value

        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as handle:
                    data = json.load(handle)
                    if isinstance(data, dict):
                        self._merge_dicts(config, data)
            except Exception as exc:
                print(f"Warning: Could not load config: {exc}")

        return config

    def _merge_dicts(self, base: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively merge updates into base dictionary"""
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._merge_dicts(base[key], value)
            else:
                base[key] = value

    def _ensure_directories(self):
        """Ensure required directories exist
        
        Note: This method caches Path objects in the config for performance,
        converting string paths to Path objects in place.
        """
        # Cache Path objects to avoid repeated conversions
        paths_config = self._config.get('paths', {})
        for path_key in ['models_dir', 'temp_dir', 'output_dir']:
            path_value = paths_config.get(path_key)
            if path_value:
                # Convert to Path object if needed and cache for future use
                if not isinstance(path_value, Path):
                    path_value = Path(path_value)
                    paths_config[path_key] = path_value  # Cache the Path object
                path_value.mkdir(parents=True, exist_ok=True)

    def _apply_dynamic_processing_defaults(self):
        """Derive processing settings that depend on current system resources."""
        processing = self._config.get('processing') or {}
        if not processing.get('auto_chunk'):
            return

        # Check if we've already computed this and memory state hasn't changed significantly
        if self._chunk_size_cache is not None and self._cached_memory_state is not None and PSUTIL_AVAILABLE:
            try:
                current_mem = psutil.virtual_memory().available
                if abs(current_mem - self._cached_memory_state) < MEMORY_CHANGE_THRESHOLD_BYTES:
                    processing['chunk_size'] = self._chunk_size_cache
                    return
            except Exception:
                pass

        min_chunk = int(processing.get('min_chunk', 80000))
        max_chunk = int(processing.get('max_chunk', max(min_chunk, processing.get('chunk_size', 160000))))
        if max_chunk < min_chunk:
            max_chunk = min_chunk

        recommendation = self._compute_auto_chunk_size(
            min_chunk=min_chunk,
            max_chunk=max_chunk,
            quality=processing.get('quality', 'balanced'),
            headroom_ratio=float(processing.get('memory_headroom_ratio', 0.65) or 0.65),
            failsafe_threshold=float(processing.get('failsafe_threshold_gb', 2.0) or 2.0)
        )

        if recommendation is not None:
            processing['chunk_size'] = recommendation
            self._chunk_size_cache = recommendation
            if PSUTIL_AVAILABLE:
                try:
                    self._cached_memory_state = psutil.virtual_memory().available
                except Exception:
                    pass

    def _compute_auto_chunk_size(
        self,
        *,
        min_chunk: int,
        max_chunk: int,
        quality: str,
        headroom_ratio: float,
        failsafe_threshold: float
    ) -> Optional[int]:
        """Compute a chunk size based on system RAM and quality preference."""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            virtual_mem = psutil.virtual_memory()
        except Exception:
            return None

        total_gb = max(0.0, virtual_mem.total / (1024 ** 3))
        available_gb = max(0.0, virtual_mem.available / (1024 ** 3))

        if available_gb <= failsafe_threshold:
            return min_chunk

        usable_gb = max(0.0, (available_gb - failsafe_threshold) * max(0.0, min(headroom_ratio, 1.0)))
        baseline = max(0.1, total_gb * max(0.0, min(headroom_ratio, 1.0)))
        utilization = min(1.0, usable_gb / baseline) if baseline else 0.0

        base_chunk = int(min_chunk + (max_chunk - min_chunk) * utilization)
        quality_multiplier = self._quality_multiplier(quality)
        adjusted_chunk = int(base_chunk * quality_multiplier)

        return max(min_chunk, min(max_chunk, adjusted_chunk))

    @staticmethod
    def _quality_multiplier(quality: str) -> float:
        """Translate quality keyword to a chunk size multiplier."""
        quality_map = {
            'eco': 0.85,
            'balanced': 1.0,
            'hq': 1.1,
            'max': 1.2
        }
        return quality_map.get(str(quality).lower(), 1.0)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'ui.theme')"""
        keys = key.split('.')
        value: Any = self._config

        for part in keys:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        target = self._config

        for part in keys[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        target[keys[-1]] = value

        # Only invalidate cache and recompute for specific processing keys
        if keys[0] == 'processing' and keys[-1] in {
            'auto_chunk',
            'quality',
            'memory_headroom_ratio',
            'min_chunk',
            'max_chunk',
            'failsafe_threshold_gb'
        }:
            # Invalidate cache
            self._chunk_size_cache = None
            self._cached_memory_state = None
            self._apply_dynamic_processing_defaults()

    def save(self):
        """Save configuration to file"""
        try:
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as handle:
                json.dump(self._config, handle, indent=2)
        except Exception as exc:
            print(f"Warning: Could not save config: {exc}")

    def reset(self):
        """Reset configuration to defaults"""
        # Use the same optimized copying as _load_config
        config = {}
        for key, value in self.DEFAULT_CONFIG.items():
            if isinstance(value, dict):
                config[key] = copy.deepcopy(value)
            else:
                config[key] = value
        self._config = config
        # Invalidate cache
        self._chunk_size_cache = None
        self._cached_memory_state = None
        self._apply_dynamic_processing_defaults()
        self.save()
