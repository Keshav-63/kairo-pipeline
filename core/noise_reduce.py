# kairo_pipeline/core/noise_reduce.py
import os
import subprocess
import shutil
from config.logger import logger

def denoise_audio(input_path, output_path):
    """
    Denoise an audio file using the Demucs model.
    """
    temp_dir = f"temp_demucs_{os.path.basename(input_path)}"
    logger.info(f"--- Starting ADVANCED noise reduction for: {input_path} ---")
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Using a more robust command structure
        cmd = [
            "python", "-m", "demucs",
            f'"{input_path}"',
            "--out", f'"{temp_dir}"',
            "--two-stems", "vocals",
            "-n", "htdemucs"
        ]
        
        logger.info(f"Running Demucs command: {' '.join(cmd)}")
        process = subprocess.run(" ".join(cmd), shell=True, check=True, text=True, capture_output=True)
        logger.info("Demucs process completed.")

        input_filename_stem = os.path.basename(input_path).rsplit('.', 1)[0]
        expected_vocals_path = os.path.join(temp_dir, "htdemucs", input_filename_stem, "vocals.wav")

        if not os.path.exists(expected_vocals_path):
            logger.error(f"Could not locate 'vocals.wav' output from Demucs in {expected_vocals_path}")
            return False
        
        logger.info(f"Found denoised audio at: {expected_vocals_path}")
        shutil.move(expected_vocals_path, output_path)
        
        logger.info(f"--- Successfully denoised {input_path} -> {output_path} ---")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Demucs subprocess failed for {input_path}. Return code: {e.returncode}")
        logger.error(f"Stderr from Demucs: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Denoising failed for {input_path}: {e}", exc_info=True)
        return False
    finally:
        if os.path.exists(temp_dir):
            logger.debug(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)