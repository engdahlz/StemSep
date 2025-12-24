"""
GPU detection and CUDA support for StemSep
"""

import platform
import psutil
import logging
from typing import Dict, List, Optional

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class GPUDetector:
    """Detects and reports GPU capabilities"""

    def __init__(self):
        self.logger = logging.getLogger('StemSep')
        self.gpu_info = None
        self._detect_gpus()

    def _detect_gpus(self):
        """Detect available GPUs"""
        self.gpu_info = {
            'has_nvidia': False,
            'has_amd': False,
            'has_intel': False,
            'has_cuda': False,
            'cuda_version': None,
            'gpus': [],
            'system_info': self._get_system_info()
        }

        # Check for CUDA support first (most reliable)
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.gpu_info['has_cuda'] = True
                self.gpu_info['cuda_version'] = torch.version.cuda
                self.gpu_info['has_nvidia'] = True

                # Get GPU details from torch (primary source)
                for i in range(torch.cuda.device_count()):
                    gpu = torch.cuda.get_device_properties(i)
                    vram_gb = gpu.total_memory / (1024**3)
                    gpu_info = {
                        'name': gpu.name,
                        'index': i,
                        'memory_gb': round(vram_gb, 2),
                        'compute_capability': f"{gpu.major}.{gpu.minor}",
                        'multiprocessors': gpu.multi_processor_count,
                        'recommended': self._is_recommended(vram_gb)
                    }
                    self.gpu_info['gpus'].append(gpu_info)

        # Only use GPUtil as fallback if torch didn't find GPUs
        if GPU_AVAILABLE and not self.gpu_info['gpus']:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info = {
                        'name': gpu.name,
                        'index': gpu.id,
                        'memory_gb': round(gpu.memoryTotal / 1024, 2),
                        'memory_free_gb': round(gpu.memoryFree / 1024, 2),
                        'memory_used_gb': round(gpu.memoryUsed / 1024, 2),
                        'load': gpu.load * 100,
                        'temperature': gpu.temperature,
                        'recommended': gpu.memoryTotal >= 6000  # 6GB minimum
                    }
                    self.gpu_info['gpus'].append(gpu_info)
                    self.gpu_info['has_nvidia'] = True
            except Exception as e:
                self.logger.warning(f"GPUtil detection failed: {e}")

        # Detect AMD/Intel GPUs only if no NVIDIA GPUs found (limited support)
        if not self.gpu_info['has_nvidia'] and platform.system() == 'Windows':
            try:
                processor_info = platform.processor()
                self.gpu_info['has_amd'] = 'AMD' in processor_info
                self.gpu_info['has_intel'] = 'Intel' in processor_info
            except Exception as e:
                self.logger.debug(f"GPU vendor detection: {e}")

    def _is_recommended(self, vram_gb: float) -> bool:
        """Check if GPU is recommended for stem separation"""
        # 6GB is the practical minimum for decent performance (MDX/Mel-Roformer)
        # 8GB+ is needed for BS-Roformer
        return vram_gb >= 6

    def _get_system_info(self) -> Dict:
        """Get system information"""
        cpu_name = platform.processor()
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            if 'brand_raw' in info:
                cpu_name = info['brand_raw']
        except Exception as e:
            self.logger.warning(f"Failed to get detailed CPU info: {e}")

        return {
            'platform': platform.platform(),
            'processor': cpu_name,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': platform.python_version(),
            'torch_available': TORCH_AVAILABLE
        }

    def get_gpu_info(self) -> Dict:
        """Get detected GPU information"""
        return self.gpu_info

    def get_best_model_type(self) -> Optional[str]:
        """Recommend best model type based on GPU capabilities"""
        if not self.gpu_info['gpus']:
            return "CPU-optimized (very slow)"

        best_gpu = max(self.gpu_info['gpus'], key=lambda x: x.get('memory_gb', 0))
        vram = best_gpu.get('memory_gb', 0)

        if vram >= 8:
            return "BS-Roformer (best quality)"
        elif vram >= 6:
            return "Mel-Roformer (good quality)"
        elif vram >= 4:
            return "MDX23C (fast)"
        else:
            return "Lightweight models only"

    def print_gpu_report(self):
        """Print a detailed GPU report"""
        print("\n" + "="*60)
        print("GPU DETECTION REPORT")
        print("="*60)

        print(f"\nSystem: {self.gpu_info['system_info']['platform']}")
        print(f"CPU: {self.gpu_info['system_info']['processor']}")
        print(f"CPU Cores: {self.gpu_info['system_info']['cpu_count']}")
        print(f"RAM: {self.gpu_info['system_info']['memory_total_gb']} GB")

        print(f"\nPyTorch: {'✓' if TORCH_AVAILABLE else '✗'}")
        if self.gpu_info['has_cuda']:
            print(f"CUDA: ✓ (v{self.gpu_info['cuda_version']})")
        else:
            print("CUDA: ✗")

        print(f"\nGPUs Detected: {len(self.gpu_info['gpus'])}")

        for i, gpu in enumerate(self.gpu_info['gpus'], 1):
            print(f"\n  GPU {i}: {gpu['name']}")
            print(f"    Memory: {gpu.get('memory_gb', 'N/A')} GB")
            if 'memory_free_gb' in gpu:
                print(f"    Free Memory: {gpu['memory_free_gb']} GB")
            if 'load' in gpu:
                print(f"    Load: {gpu['load']:.1f}%")
            if 'multiprocessors' in gpu:
                print(f"    SMs: {gpu['multiprocessors']}")
            print(f"    Recommended: {'✓' if gpu['recommended'] else '✗'}")

        best_model = self.get_best_model_type()
        print(f"\nRecommended Model Type: {best_model}")
        print("="*60 + "\n")

# Test function
if __name__ == "__main__":
    detector = GPUDetector()
    detector.print_gpu_report()
