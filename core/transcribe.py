import whisper
import numpy as np
from config.logger import logger
import soundfile as sf # Added to check sampling rate

def transcribe_audio(audio_data: np.ndarray, whisper_model, audio_sr: int = 16000):
    """
    Transcribes a NumPy array of audio data using a pre-loaded Whisper model,
    ensuring the output is always translated to English, returns rich metadata
    including word-level timestamps, and corrects 'Cairo' to 'Kairo'.
    """
    logger.info("--- Starting transcription with Whisper model (Task: Translate to English) ---")
    try:
        # --- DEFINITIVE FIX: Ensure audio data is float32 and correct sampling rate ---
        if audio_data.dtype != np.float32:
            audio_data_float32 = audio_data.astype(np.float32)
        else:
            audio_data_float32 = audio_data

        # Whisper's default model expects 16kHz audio.
        # Ensure the `audio_data` is a 1D array.
        if audio_data_float32.ndim > 1:
            audio_data_float32 = audio_data_float32.mean(axis=1) # Convert to mono if stereo

        # Use the "translate" task to ensure English output and word timestamps
        result = whisper_model.transcribe(
            audio_data_float32, 
            task="translate", 
            word_timestamps=True,
            verbose=False 
        )
        
        # --- NEW: Apply Kairo correction ---
        if result and 'text' in result and result['text']:
            original_text = result['text']
            corrected_text = original_text.replace("Cairo", "Kairo").replace("cairo", "Kairo").replace("CAIRO", "Kairo")
            if original_text != corrected_text:
                 logger.debug(f"Applied Kairo correction within transcribe_audio. Original: '{original_text[:50]}...' Corrected: '{corrected_text[:50]}...'")
                 result['text'] = corrected_text # Update the result dictionary
        # --- END NEW ---
        
        logger.info("Transcription completed successfully.")  
        # The result dictionary now contains the corrected 'text' and 'segments' with 'words'.
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return None
