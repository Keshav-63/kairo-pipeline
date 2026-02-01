# kairo_pipeline/core/preprocess.py
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize
from config.logger import logger
import os
# Removed: import subprocess (no longer needed for Demucs)
# Removed: import shutil (no longer needed for Demucs temp directories)


def preprocess_audio(input_path, output_path, target_sr=16000):
    """
    Preprocesses an audio file by converting it to 16kHz mono,
    normalizing it, and applying spectral gating noise reduction.
    The Demucs process has been removed for efficiency.
    """
    logger.info(f"--- Starting preprocessing for '{input_path}' (Demucs disabled) ---")
    
    # No intermediate_path needed if Demucs is removed.
    # The final_output_path is simply the provided output_path.

    try:
        # Step 1: Load and convert audio to 16kHz mono
        logger.info("Step 1: Loading and converting audio to 16kHz mono...")
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(target_sr)
        audio = audio.set_channels(1)
        
        # Step 2: Normalizing audio volume
        logger.info("Step 2: Normalizing audio volume...")
        normalized_audio = normalize(audio)

        # Step 3: Convert to numpy array for noise reduction
        logger.info("Step 3: Converting to numpy array for processing...")
        samples = np.array(normalized_audio.get_array_of_samples()).astype(np.float32) / (2**15)

        # Step 4: Applying spectral gating noise reduction
        logger.info("Step 4: Applying spectral gating noise reduction...")
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=target_sr, stationary=True)

        # Ensure the final output is np.float32 for consistency
        final_audio_data = reduced_noise_samples.astype(np.float32)

        logger.info(f"Step 5: Saving preprocessed file to '{output_path}'...")
        sf.write(output_path, final_audio_data, target_sr)
        
        logger.info(f"--- Preprocessing complete for '{input_path}'. Output at '{output_path}' ---")
        return True

    except Exception as e:
        logger.error(f"An error occurred during preprocessing for '{input_path}': {e}", exc_info=True)
        return False
