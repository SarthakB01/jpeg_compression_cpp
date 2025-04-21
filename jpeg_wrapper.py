import subprocess
import os
from pathlib import Path

def compress_jpeg(input_path, output_path, quality):
    try:
        # Get path to our renamed executable
        executable = Path(__file__).parent / "jpeg_compressor"
        
        # Verify the executable exists
        if not executable.exists():
            raise FileNotFoundError(f"Compressor executable not found at {executable}")
        
        # Set execute permissions (important!)
        os.chmod(executable, 0o755)
        
        # Call our renamed executable
        result = subprocess.run(
            [str(executable), input_path, output_path, str(quality)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
            
    except Exception as e:
        return False, f"Compression failed: {str(e)}"