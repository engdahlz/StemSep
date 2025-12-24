"""
System diagnostics module for checking hardware and software capabilities.

This module provides comprehensive system information including GPU, memory,
CPU, disk space, and software versions.
"""

import logging
import platform
import psutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SystemDiagnostics:
    """
    Comprehensive system diagnostics for audio separation application.
    
    Provides methods to check CUDA/GPU availability, memory, CPU, disk space,
    and other system information relevant for running ML models.
    """
    
    def __init__(self):
        """Initialize SystemDiagnostics."""
        self._torch = None
        self._cuda_available = None
        
    def _import_torch(self) -> bool:
        """
        Lazy import of torch to avoid dependency issues.
        
        Returns:
            bool: True if torch is available, False otherwise.
        """
        if self._torch is None:
            try:
                import torch
                self._torch = torch
                return True
            except ImportError:
                logger.warning("PyTorch not installed. GPU features unavailable.")
                return False
        return True
    
    def check_cuda_availability(self) -> Dict[str, Any]:
        """
        Check CUDA/GPU availability and details.
        
        Returns:
            dict: Dictionary containing:
                - available (bool): Whether CUDA is available
                - gpu_count (int): Number of GPUs detected
                - cuda_version (str|None): CUDA version if available
                - gpus (list): List of GPU details
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> cuda_info = diagnostics.check_cuda_availability()
            >>> if cuda_info['available']:
            ...     print(f"Found {cuda_info['gpu_count']} GPU(s)")
        """
        logger.debug("Checking CUDA availability...")
        
        result = {
            "available": False,
            "gpu_count": 0,
            "cuda_version": None,
            "gpus": []
        }
        
        if not self._import_torch():
            return result
        
        try:
            result["available"] = self._torch.cuda.is_available()
            
            if result["available"]:
                result["gpu_count"] = self._torch.cuda.device_count()
                result["cuda_version"] = self._torch.version.cuda
                
                # Get details for each GPU
                for i in range(result["gpu_count"]):
                    gpu_info = {
                        "id": i,
                        "name": self._torch.cuda.get_device_name(i),
                        "compute_capability": self._torch.cuda.get_device_capability(i),
                        "total_memory_gb": round(
                            self._torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                        )
                    }
                    result["gpus"].append(gpu_info)
                
                logger.info(f"CUDA available: {result['gpu_count']} GPU(s) detected")
            else:
                logger.info("CUDA not available - CPU mode only")
                
        except Exception as e:
            logger.error(f"Error checking CUDA: {e}")
            result["error"] = str(e)
        
        return result
    
    def get_vram_info(self) -> Dict[str, Any]:
        """
        Get detailed VRAM information for all GPUs.
        
        Returns:
            dict: Dictionary containing:
                - available (bool): Whether GPU is available
                - gpus (list): List of VRAM info per GPU with:
                    - id (int): GPU index
                    - name (str): GPU name
                    - total_gb (float): Total VRAM in GB
                    - allocated_gb (float): Currently allocated VRAM in GB
                    - reserved_gb (float): Reserved VRAM in GB
                    - free_gb (float): Free VRAM in GB
                    
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> vram = diagnostics.get_vram_info()
            >>> for gpu in vram['gpus']:
            ...     print(f"{gpu['name']}: {gpu['free_gb']:.1f} GB free")
        """
        logger.debug("Getting VRAM information...")
        
        result = {
            "available": False,
            "gpus": []
        }
        
        if not self._import_torch():
            return result
        
        try:
            if not self._torch.cuda.is_available():
                logger.debug("CUDA not available - no VRAM info")
                return result
            
            result["available"] = True
            
            for i in range(self._torch.cuda.device_count()):
                total = self._torch.cuda.get_device_properties(i).total_memory
                allocated = self._torch.cuda.memory_allocated(i)
                reserved = self._torch.cuda.memory_reserved(i)
                free = total - reserved
                
                gpu_vram = {
                    "id": i,
                    "name": self._torch.cuda.get_device_name(i),
                    "total_gb": round(total / (1024**3), 2),
                    "allocated_gb": round(allocated / (1024**3), 2),
                    "reserved_gb": round(reserved / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2)
                }
                result["gpus"].append(gpu_vram)
                
                logger.debug(f"GPU {i}: {gpu_vram['free_gb']:.1f} GB free")
            
        except Exception as e:
            logger.error(f"Error getting VRAM info: {e}")
            result["error"] = str(e)
        
        return result
    
    def get_system_memory(self) -> Dict[str, Any]:
        """
        Get system RAM information.
        
        Returns:
            dict: Dictionary containing:
                - total_gb (float): Total RAM in GB
                - available_gb (float): Available RAM in GB
                - used_gb (float): Used RAM in GB
                - percent_used (float): Percentage of RAM used
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> ram = diagnostics.get_system_memory()
            >>> print(f"RAM: {ram['available_gb']:.1f} GB available")
        """
        logger.debug("Getting system memory info...")
        
        try:
            mem = psutil.virtual_memory()
            
            result = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "percent_used": mem.percent
            }
            
            logger.debug(f"RAM: {result['available_gb']:.1f} GB available")
            return result
            
        except Exception as e:
            logger.error(f"Error getting system memory: {e}")
            return {"error": str(e)}
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information.
        
        Returns:
            dict: Dictionary containing:
                - physical_cores (int): Number of physical CPU cores
                - logical_cores (int): Number of logical CPU cores (with hyperthreading)
                - max_frequency_mhz (float): Max CPU frequency in MHz
                - current_frequency_mhz (float): Current CPU frequency in MHz
                - cpu_percent (float): Current CPU usage percentage
                - model (str): CPU model name
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> cpu = diagnostics.get_cpu_info()
            >>> print(f"CPU: {cpu['physical_cores']} cores")
        """
        logger.debug("Getting CPU info...")
        
        try:
            result = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "model": platform.processor() or "Unknown"
            }
            
            # Get frequency info (may not be available on all systems)
            try:
                freq = psutil.cpu_freq()
                if freq:
                    result["max_frequency_mhz"] = round(freq.max, 2)
                    result["current_frequency_mhz"] = round(freq.current, 2)
            except Exception as e:
                logger.debug(f"CPU frequency info not available: {e}")
            
            # Get current CPU usage
            result["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            logger.debug(f"CPU: {result['physical_cores']} physical cores")
            return result
            
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}")
            return {"error": str(e)}
    
    def get_disk_space(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get disk space information.
        
        Args:
            path: Path to check disk space for. Defaults to current directory.
            
        Returns:
            dict: Dictionary containing:
                - path (str): Path checked
                - total_gb (float): Total disk space in GB
                - used_gb (float): Used disk space in GB
                - free_gb (float): Free disk space in GB
                - percent_used (float): Percentage of disk used
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> disk = diagnostics.get_disk_space("/home/user/music")
            >>> print(f"Disk: {disk['free_gb']:.1f} GB free")
        """
        if path is None:
            path = str(Path.cwd())
        
        logger.debug(f"Getting disk space for: {path}")
        
        try:
            usage = psutil.disk_usage(path)
            
            result = {
                "path": path,
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "percent_used": usage.percent
            }
            
            logger.debug(f"Disk: {result['free_gb']:.1f} GB free")
            return result
            
        except Exception as e:
            logger.error(f"Error getting disk space: {e}")
            return {"error": str(e), "path": path}
    
    def get_os_info(self) -> Dict[str, str]:
        """
        Get operating system information.
        
        Returns:
            dict: Dictionary containing:
                - system (str): OS name (Linux, Windows, Darwin)
                - release (str): OS release version
                - version (str): OS version details
                - machine (str): Machine type (x86_64, etc)
                -
 (str): Platform string
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> os_info = diagnostics.get_os_info()
            >>> print(f"OS: {os_info['system']} {os_info['release']}")
        """
        logger.debug("Getting OS info...")
        
        try:
            return {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "platform": platform.platform()
            }
        except Exception as e:
            logger.error(f"Error getting OS info: {e}")
            return {"error": str(e)}
    
    def get_python_info(self) -> Dict[str, str]:
        """
        Get Python version and environment information.
        
        Returns:
            dict: Dictionary containing:
                - version (str): Python version
                - implementation (str): Python implementation (CPython, PyPy, etc)
                - executable (str): Path to Python executable
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> py_info = diagnostics.get_python_info()
            >>> print(f"Python: {py_info['version']}")
        """
        logger.debug("Getting Python info...")
        
        try:
            return {
                "version": sys.version.split()[0],
                "implementation": platform.python_implementation(),
                "executable": sys.executable
            }
        except Exception as e:
            logger.error(f"Error getting Python info: {e}")
            return {"error": str(e)}
    
    def get_dependencies_info(self) -> Dict[str, str]:
        """
        Get versions of key dependencies.
        
        Returns:
            dict: Dictionary with package names as keys and versions as values.
                Includes torch, torchaudio, librosa, soundfile, psutil, etc.
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> deps = diagnostics.get_dependencies_info()
            >>> print(f"PyTorch: {deps.get('torch', 'Not installed')}")
        """
        logger.debug("Getting dependencies info...")
        
        dependencies = {}
        
        # List of packages to check
        packages = [
            "torch",
            "torchaudio", 
            "librosa",
            "soundfile",
            "numpy",
            "scipy",
            "psutil",
            "customtkinter",
            "pillow"
        ]
        
        for package in packages:
            try:
                if package == "torch" and self._torch:
                    dependencies[package] = self._torch.__version__
                else:
                    module = __import__(package)
                    dependencies[package] = getattr(module, "__version__", "Unknown")
            except ImportError:
                dependencies[package] = "Not installed"
            except Exception as e:
                dependencies[package] = f"Error: {e}"
        
        return dependencies
    
    def get_full_system_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive system report with all diagnostics.
        
        Returns:
            dict: Complete system report including:
                - cuda: CUDA/GPU information
                - vram: VRAM details per GPU
                - memory: System RAM information
                - cpu: CPU details
                - disk: Disk space information
                - os: Operating system information
                - python: Python version and environment
                - dependencies: Installed package versions
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> report = diagnostics.get_full_system_report()
            >>> print(f"System ready: {report['cuda']['available']}")
        """
        logger.info("Generating full system report...")
        
        report = {
            "cuda": self.check_cuda_availability(),
            "vram": self.get_vram_info(),
            "memory": self.get_system_memory(),
            "cpu": self.get_cpu_info(),
            "disk": self.get_disk_space(),
            "os": self.get_os_info(),
            "python": self.get_python_info(),
            "dependencies": self.get_dependencies_info()
        }
        
        logger.info("System report generated successfully")
        return report
    
    def can_run_model(self, required_vram_gb: float) -> Dict[str, Any]:
        """
        Check if system can run a model with given VRAM requirements.
        
        Args:
            required_vram_gb: Required VRAM in GB.
            
        Returns:
            dict: Dictionary containing:
                - can_run (bool): Whether model can run
                - reason (str): Explanation
                - available_vram_gb (float): Available VRAM
                - gpu_name (str|None): GPU name if available
                
        Raises:
            ValueError: If required_vram_gb is negative.
            
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> result = diagnostics.can_run_model(8.0)
            >>> if result['can_run']:
            ...     print("Model can run!")
        """
        if required_vram_gb < 0:
            raise ValueError("required_vram_gb must be non-negative")
        
        logger.debug(f"Checking if system can run model requiring {required_vram_gb} GB VRAM")
        
        vram_info = self.get_vram_info()
        
        if not vram_info["available"]:
            return {
                "can_run": False,
                "reason": "No GPU available - CPU mode only (slower)",
                "available_vram_gb": 0.0,
                "gpu_name": None
            }
        
        # Check if any GPU has enough free VRAM
        for gpu in vram_info["gpus"]:
            if gpu["free_gb"] >= required_vram_gb:
                return {
                    "can_run": True,
                    "reason": f"Sufficient VRAM available on {gpu['name']}",
                    "available_vram_gb": gpu["free_gb"],
                    "gpu_name": gpu["name"]
                }
        
        # Find GPU with most VRAM
        best_gpu = max(vram_info["gpus"], key=lambda g: g["free_gb"])
        
        return {
            "can_run": False,
            "reason": f"Insufficient VRAM. Need {required_vram_gb} GB, have {best_gpu['free_gb']} GB free",
            "available_vram_gb": best_gpu["free_gb"],
            "gpu_name": best_gpu["name"]
        }
    
    def get_recommended_gpu_profile(self) -> Dict[str, Any]:
        """
        Get recommended GPU profile based on detected hardware.
        
        Uses GPU_PROFILES from model_config to return optimal settings.
        
        Returns:
            dict: Dictionary containing:
                - profile_name (str): Name of recommended profile
                - settings (dict): Profile settings (batch_size, use_amp, etc.)
                - gpu_info (dict): Detected GPU information
                - vram_gb (float): Available VRAM
                
        Example:
            >>> diagnostics = SystemDiagnostics()
            >>> profile = diagnostics.get_recommended_gpu_profile()
            >>> print(f"Using profile: {profile['profile_name']}")
        """
        logger.debug("Detecting recommended GPU profile...")
        
        # Get GPU info from CUDA first
        cuda_info = self.check_cuda_availability()
        
        result = {
            "profile_name": "cpu",
            "settings": {},
            "gpu_info": None,
            "vram_gb": 0.0
        }
        
        # Import GPU_PROFILES from model_config
        # Use try/except with both absolute and relative import for compatibility
        try:
            # Absolute import (works when sys.path includes StemSepApp/src)
            from models.model_config import GPU_PROFILES, get_gpu_settings
        except ImportError:
            try:
                # Relative import (works when running as part of package)
                from ..models.model_config import GPU_PROFILES, get_gpu_settings
            except ImportError:
                logger.warning("Could not import GPU_PROFILES from model_config")
                return result
        
        # Try to get GPU info from GPUDetector as fallback if CUDA not available
        gpu_info = None
        vram_gb = 0.0
        
        if cuda_info["available"] and cuda_info["gpus"]:
            # Use CUDA info directly
            gpu_info = cuda_info["gpus"][0]
            vram_gb = gpu_info.get("total_memory_gb", 0)
        else:
            # Fallback: Try GPUDetector which uses GPUtil
            try:
                from .gpu_detector import GPUDetector
                detector = GPUDetector()
                detector_info = detector.get_gpu_info()
                if detector_info.get("gpus"):
                    # Found GPU via GPUtil even though CUDA not available
                    gpu = detector_info["gpus"][0]
                    gpu_info = {
                        "name": gpu.get("name", "Unknown GPU"),
                        "total_memory_gb": gpu.get("memory_gb", 0)
                    }
                    vram_gb = gpu.get("memory_gb", 0)
                    logger.info(f"Using GPUDetector fallback: {gpu_info['name']} with {vram_gb} GB")
            except Exception as e:
                logger.debug(f"GPUDetector fallback failed: {e}")
        
        if not gpu_info:
            result["settings"] = GPU_PROFILES.get("cpu", {})
            logger.info("No GPU detected - using CPU profile")
            return result
        
        result["gpu_info"] = gpu_info
        result["vram_gb"] = vram_gb
        gpu_name = gpu_info.get("name", "").lower()
        
        # Detect GPU type
        if "nvidia" in gpu_name or "geforce" in gpu_name or "rtx" in gpu_name or "gtx" in gpu_name:
            # NVIDIA GPU
            if "gtx 10" in gpu_name or "gtx10" in gpu_name:
                profile_name = "nvidia_gtx10xx"
            else:
                profile_name = "nvidia_cuda"
        elif "amd" in gpu_name or "radeon" in gpu_name:
            # AMD GPU - check for ROCm (Linux) vs DirectML (Windows)
            import platform
            if platform.system() == "Linux":
                profile_name = "amd_rocm"
            else:
                profile_name = "amd_directml"
        elif "intel" in gpu_name or "arc" in gpu_name:
            profile_name = "intel_openvino"
        else:
            profile_name = "nvidia_cuda"  # Default to CUDA
        
        result["profile_name"] = profile_name
        result["settings"] = get_gpu_settings(profile_name, vram_gb)
        
        logger.info(f"Recommended GPU profile: {profile_name} (VRAM: {vram_gb} GB)")
        return result


if __name__ == "__main__":
    # Quick test/demo
    logging.basicConfig(level=logging.INFO)
    
    diag = SystemDiagnostics()
    report = diag.get_full_system_report()
    
    print("\n=== System Diagnostics Report ===\n")
    
    print(f"OS: {report['os']['system']} {report['os']['release']}")
    print(f"Python: {report['python']['version']}")
    print(f"\nCPU: {report['cpu']['physical_cores']} cores ({report['cpu']['logical_cores']} logical)")
    print(f"RAM: {report['memory']['available_gb']:.1f} GB available / {report['memory']['total_gb']:.1f} GB total")
    
    if report['cuda']['available']:
        print(f"\nGPU: {report['cuda']['gpu_count']} GPU(s) detected")
        for gpu in report['cuda']['gpus']:
            print(f"  - {gpu['name']}: {gpu['total_memory_gb']} GB VRAM")
    else:
        print("\nGPU: Not available (CPU mode only)")
    
    print(f"\nDisk: {report['disk']['free_gb']:.1f} GB free / {report['disk']['total_gb']:.1f} GB total")
    
    print("\nKey Dependencies:")
    for pkg, ver in report['dependencies'].items():
        print(f"  - {pkg}: {ver}")
