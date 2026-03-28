
# collect_env_info.py
import os
import sys
import platform
import subprocess
import shutil
import re
from datetime import datetime

def try_import(name):
    try:
        mod = __import__(name)
        return mod, getattr(mod, "__version__", "unknown")
    except Exception as e:
        return None, f"NOT INSTALLED ({e.__class__.__name__})"

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, shell=False)
        return out.strip()
    except Exception as e:
        return f"FAILED ({e.__class__.__name__}: {e})"

def get_cpu_name():
    try:
        if sys.platform.startswith("win"):
            # WMIC is deprecated but still available on many systems
            out = run_cmd(["wmic", "cpu", "get", "name"])
            lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
            return lines[0] if lines else platform.processor()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            m = re.search(r"model name\s*:\s*(.+)", txt)
            return m.group(1).strip() if m else platform.processor()
        elif sys.platform.startswith("darwin"):
            out = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
            return out if not out.startswith("FAILED") else platform.processor()
        else:
            return platform.processor()
    except Exception:
        return platform.processor()

def get_total_ram_gb():
    # Try psutil first
    psutil, ver = try_import("psutil")
    if psutil:
        try:
            return round(psutil.virtual_memory().total / (1024**3), 2)
        except Exception:
            pass

    # Fallbacks
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                meminfo = f.read()
            m = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
            if m:
                kb = int(m.group(1))
                return round(kb * 1024 / (1024**3), 2)
        elif sys.platform.startswith("darwin"):
            out = run_cmd(["sysctl", "-n", "hw.memsize"])
            if out.isdigit():
                return round(int(out) / (1024**3), 2)
        elif sys.platform.startswith("win"):
            out = run_cmd(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])
            nums = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
            if nums:
                return round(nums[0] / (1024**3), 2)
    except Exception:
        pass
    return "unknown"

def get_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return None
    return run_cmd(["nvidia-smi"])

def main():
    print("=== Experiment Environment Info ===")
    print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    # OS / Python
    print("---- OS / Python ----")
    print("OS:", platform.system(), platform.release())
    print("OS version:", platform.version())
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    print("Python:", sys.version.replace("\n", " "))
    print("Python executable:", sys.executable)
    print()

    # CPU / RAM
    print("---- CPU / RAM ----")
    print("CPU:", get_cpu_name())
    print("Logical cores:", os.cpu_count())
    print("Total RAM (GB):", get_total_ram_gb())
    print()

    # Key Python libs
    print("---- Key Python Packages ----")
    for pkg in ["numpy", "scipy", "pandas", "cv2", "skimage", "tqdm"]:
        mod, ver = try_import(pkg)
        # OpenCV version lives in cv2.__version__
        if pkg == "cv2" and mod is not None:
            ver = getattr(mod, "__version__", ver)
        print(f"{pkg}: {ver}")
    print()

    # Torch / CUDA
    print("---- PyTorch / CUDA ----")
    torch, torch_ver = try_import("torch")
    print("torch:", torch_ver)
    if torch is None:
        print("PyTorch not available; skipping CUDA/CUDNN details.")
        print()
    else:
        try:
            print("torch.cuda.is_available:", torch.cuda.is_available())
            print("torch.version.cuda:", torch.version.cuda)
            print("torch.backends.cudnn.version:", torch.backends.cudnn.version())
            print("torch.backends.cudnn.enabled:", torch.backends.cudnn.enabled)
            print("torch.backends.cudnn.benchmark:", torch.backends.cudnn.benchmark)
            print("torch.backends.cudnn.deterministic:", torch.backends.cudnn.deterministic)
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                print("GPU count:", n)
                for i in range(n):
                    name = torch.cuda.get_device_name(i)
                    cap = torch.cuda.get_device_capability(i)
                    props = torch.cuda.get_device_properties(i)
                    print(f"GPU[{i}] name:", name)
                    print(f"GPU[{i}] capability:", cap)
                    print(f"GPU[{i}] total_memory(GB):", round(props.total_memory / (1024**3), 2))
        except Exception as e:
            print("Torch CUDA query failed:", repr(e))
        print()

    # NVIDIA driver via nvidia-smi
    print("---- NVIDIA Driver (nvidia-smi) ----")
    smi = get_nvidia_smi()
    if smi is None:
        print("nvidia-smi not found in PATH.")
    else:
        print(smi)
    print()

    # Optional: pip freeze hint
    print("---- Optional (for full reproducibility) ----")
    print("If you want a full package snapshot, run:")
    print("  pip freeze > requirements_freeze.txt")
    print("and paste the first ~30 lines + key packages if needed.")

if __name__ == "__main__":
    main()
