import subprocess
import os
import tempfile

def compress_jpeg(input_path, output_path, quality):
    """
    Wrapper to call the C++ JPEG compression program
    
    Args:
        input_path: Path to input JPEG file
        output_path: Path to save compressed JPEG file
        quality: Compression quality (0.002-1.0)
        
    Returns:
        bool: True if compression succeeded, False otherwise
    """
    try:
        # Call the C++ executable
        result = subprocess.run(
            ["./jpeg-cpp", input_path, output_path, str(quality)],
            capture_output=True,
            text=True
        )
        
        # Check if the command was successful
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
            
    except Exception as e:
        return False, str(e)