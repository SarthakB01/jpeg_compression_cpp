# jpeg_wrapper.py (updated for Streamlit Cloud)
import subprocess
import os
import sys
from pathlib import Path

def get_platform_executable():
    """Return the correct pre-compiled executable for the platform"""
    base_path = Path(__file__).parent
    platform = sys.platform
    
    if platform == "linux":
        return base_path / "jpeg-cpp-linux"
    elif platform == "darwin":
        return base_path / "jpeg-cpp-mac"
    elif platform == "win32":
        return base_path / "jpeg-cpp-win.exe"
    else:
        raise RuntimeError(f"Unsupported platform: {platform}")

def compress_jpeg(input_path, output_path, quality):
    """Wrapper for the pre-compiled C++ executable"""
    try:
        executable = get_platform_executable()
        
        if not executable.exists():
            raise RuntimeError(f"Missing pre-compiled executable: {executable}")
        
        # Make executable if on Unix-like system
        if sys.platform != "win32":
            os.chmod(executable, 0o755)
        
        result = subprocess.run(
            [str(executable), input_path, output_path, str(quality)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
            
    except Exception as e:
        return False, str(e)