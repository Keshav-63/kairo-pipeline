
# kairo_pipeline/core/vad_transcribe.py

import soundfile as sf
import numpy as np
from config.logger import logger
from .vad import detect_voice_activity
from .transcribe import transcribe_audio

def process_and_transcribe_vad_whisper(input_path: str, vad_model, vad_utils, whisper_model, padding_ms: int = 300):
    """
    Runs VAD, concatenates voiced chunks, transcribes with Whisper, and REMAPS
    word/segment timestamps from the concatenated timeline back to the original audio timeline.
    """
    logger.info(f"--- Starting VAD & Transcribe process for '{input_path}' ---")
    try:
        speech_timestamps = detect_voice_activity(input_path, vad_model, vad_utils, padding_ms=padding_ms)
        if not speech_timestamps:
            logger.warning(f"VAD found no speech in {input_path}. Skipping transcription.")
            return None

        audio, sr = sf.read(input_path)
        if sr != 16000:
            logger.warning(f"Audio sample rate is {sr}Hz. Whisper/VAD models typically expect 16000Hz.")

        collected_chunks = []
        chunk_meta = []
        combined_samples_so_far = 0

        for ts in speech_timestamps:
            start_idx = ts['start']
            end_idx = ts['end']
            if start_idx < len(audio) and end_idx <= len(audio) and start_idx < end_idx:
                seg = audio[start_idx:end_idx]
                collected_chunks.append(seg)
                seg_len = len(seg)
                chunk_meta.append({
                    "orig_start_samp": start_idx,
                    "orig_end_samp": end_idx,
                    "orig_start_sec": start_idx / sr,
                    "orig_end_sec": end_idx / sr,
                    "combined_start_samp": combined_samples_so_far,
                    "combined_end_samp": combined_samples_so_far + seg_len,
                    "combined_start_sec": combined_samples_so_far / sr,
                    "combined_end_sec": (combined_samples_so_far + seg_len) / sr,
                })
                combined_samples_so_far += seg_len
            else:
                logger.warning(f"Invalid VAD timestamp: start={start_idx}, end={end_idx}, audio_len={len(audio)}.")

        if not collected_chunks:
            logger.warning(f"VAD detected segments but failed to collect valid audio from {input_path}.")
            return None

        combined_audio = np.concatenate(collected_chunks)
        transcription_result = transcribe_audio(combined_audio, whisper_model, sr)
        if not transcription_result or not transcription_result.get("segments"):
            logger.warning("Transcription did not produce any segments.")
            return transcription_result

        def map_combined_to_original(t_combined_sec: float) -> float:
            for meta in chunk_meta:
                if meta["combined_start_sec"] <= t_combined_sec < meta["combined_end_sec"]:
                    offset = t_combined_sec - meta["combined_start_sec"]
                    return meta["orig_start_sec"] + offset
            return chunk_meta[-1]["orig_end_sec"]

        # Remap segment and word timestamps
        for seg in transcription_result.get("segments", []):
            if "start" in seg and "end" in seg:
                seg["start"] = map_combined_to_original(seg["start"])
                seg["end"] = map_combined_to_original(seg["end"])
            if "words" in seg and isinstance(seg["words"], list):
                for w in seg["words"]:
                    if "start" in w and "end" in w:
                        w["start"] = map_combined_to_original(w["start"])
                        w["end"] = map_combined_to_original(w["end"])

        transcription_result["time_origin"] = "original"
        logger.info(f"--- VAD & Transcribe successful with original timeline remapped ---")
        return transcription_result

    except Exception as e:
        logger.error(f"VAD & Transcribe failed for '{input_path}': {e}", exc_info=True)
        return None
