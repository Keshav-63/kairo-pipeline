# kairo_pipeline/core/vad.py
import torch
from config.logger import logger

def detect_voice_activity(audio_path: str, model, utils, padding_ms: int = 300):
    """
    Detects speech timestamps in an audio file using a pre-loaded Silero VAD model,
    adding padding to the start and end of each detected speech segment.
    
    padding_ms: milliseconds of padding to add to each side of a speech segment.
    """
    logger.info(f"--- Starting Voice Activity Detection for '{audio_path}' with {padding_ms}ms padding ---")
    try:
        (get_speech_timestamps, _, read_audio, _, _) = utils
        wav = read_audio(audio_path, sampling_rate=16000)
        sampling_rate = 16000 # Hardcoded as per common practice for VAD models

        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
        
        padded_speech_timestamps = []
        for ts in speech_timestamps:
            start_sample = max(0, ts['start'] - int(padding_ms / 1000 * sampling_rate))
            end_sample = min(len(wav), ts['end'] + int(padding_ms / 1000 * sampling_rate))
            padded_speech_timestamps.append({
                'start': start_sample,
                'end': end_sample
            })
            logger.debug(f"Original: [{ts['start']},{ts['end']}] -> Padded: [{start_sample},{end_sample}]")

        logger.info(f"Found {len(speech_timestamps)} original speech segments, resulting in {len(padded_speech_timestamps)} padded segments in '{audio_path}'.")
        return padded_speech_timestamps
    except Exception as e:
        logger.error(f"Error during VAD processing for '{audio_path}': {e}", exc_info=True)
        return []