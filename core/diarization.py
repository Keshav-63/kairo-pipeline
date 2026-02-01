# kairo_pipeline/core/diarization.py

import torch
from pyannote.audio import Pipeline
from config.logger import logger
import warnings
import json
import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from .voice_recognition import VoiceRecognizer
from collections import defaultdict
import tempfile

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration constants
MIN_TURN_SEC_FOR_EMB = 1.0     # Minimum turn duration for embedding extraction
ALIGN_TOLERANCE_SEC = 0.15       # Tolerance for word-to-turn alignment
WORD_MERGE_GAP_SEC = 2.0        # Max gap to merge words into same segment


class DiarizationPipeline:
    def __init__(self, hf_token: str, **kwargs):
        """
        Initialize multi-scale diarization pipeline with voice recognition.
        """
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"DiarizationPipeline initializing on device: {self.device}")
        
        self.voice_recognizer = VoiceRecognizer()
        logger.info("VoiceRecognizer loaded successfully.")

        try:
            logger.info("Loading pyannote.audio diarization pipeline base checkpoint...")
            self.base_checkpoint = "pyannote/speaker-diarization-3.1"
            self.pipelines = {}
            
            # Multi-scale configurations for different audio lengths
            self.scale_configs = {
                "short": {
                    "segmentation": {"min_duration_off": 0.01},
                    "clustering": {"method": "centroid", "threshold": 0.60, "min_cluster_size": 5}
                },
                "medium": {
                    "segmentation": {"min_duration_off": 0.05},
                    "clustering": {"method": "centroid", "threshold": 0.65, "min_cluster_size": 8}
                },
                "long": {
                    "segmentation": {"min_duration_off": 0.20},
                    "clustering": {"method": "centroid", "threshold": 0.70, "min_cluster_size": 10}
                }
            }

            # Instantiate all scale pipelines
            for name, params in self.scale_configs.items():
                logger.info(f"Instantiating diarization pipeline for scale '{name}' with params: {params}")
                p = Pipeline.from_pretrained(
                    self.base_checkpoint,
                    use_auth_token=self.hf_token
                )
                p.instantiate(params)
                try:
                    p.to(self.device)
                    logger.debug(f"Pipeline '{name}' moved to {self.device}")
                except Exception as e:
                    logger.debug(f"pipeline.to(device) not supported for scale '{name}': {e}")
                self.pipelines[name] = p

            logger.info(f"✓ All diarization pipelines loaded for scales: {list(self.pipelines.keys())}")

        except Exception as e:
            logger.error(f"Failed to load AI models for DiarizationPipeline: {e}", exc_info=True)
            raise

    def process(self, audio_path: str, transcription_result: dict, enrollments: list = []):
        """
        Main diarization pipeline with centroid-based enrollment mapping and interval overlap alignment.
        
        Args:
            audio_path: Path to audio file
            transcription_result: Dict from Whisper with segments and words
            enrollments: List of enrolled speaker dicts with 'embedding' and 'personName'
        
        Returns:
            List of segments with speaker labels, text, start, and end times
        """
        logger.info(f"{'='*80}")
        logger.info(f"Starting Diarization Process for: {os.path.basename(audio_path)}")
        logger.info(f"Enrollments provided: {len(enrollments)}")
        logger.info(f"{'='*80}")

        try:
            # Validate transcription result
            if not transcription_result or not transcription_result.get("segments"):
                logger.warning(f"Transcription result is empty or invalid for {audio_path}")
                return []

            transcript_preview = transcription_result.get("text", "")[:200].replace('\n', ' ')
            logger.info(f"Transcription preview: {transcript_preview}...")
            
            # Get audio duration for adaptive processing
            try:
                info = sf.info(audio_path)
                duration = info.duration
                logger.info(f"Audio duration: {duration:.2f}s, Sample rate: {info.samplerate}Hz")
            except Exception as e:
                logger.warning(f"Could not read audio info: {e}. Using default duration.")
                duration = 30.0

            # STEP 1: Run multi-scale diarization
            logger.info("STEP 1: Running multi-scale pyannote diarization...")
            all_turns = []
            
            for scale_name, pipeline in self.pipelines.items():
                logger.info(f"  → Running '{scale_name}' scale diarization...")
                try:
                    diarization_result = pipeline(audio_path)
                    scale_turns = []
                    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                        scale_turns.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker,
                            "scale": scale_name,
                            "duration": turn.end - turn.start
                        })
                    logger.info(f"    ✓ '{scale_name}' found {len(scale_turns)} turns")
                    all_turns.extend(scale_turns)
                except Exception as e:
                    logger.error(f"    ✗ '{scale_name}' diarization failed: {e}", exc_info=True)
                    continue

            if not all_turns:
                logger.error("No speaker turns detected by any diarization pipeline.")
                return []
            
            logger.info(f"STEP 1 Complete: Collected {len(all_turns)} total turns from all scales")

            # STEP 2: Merge multi-scale results
            logger.info("STEP 2: Merging multi-scale diarization results...")
            speaker_turns = self.merge_diarization_results(all_turns)
            logger.info(f"STEP 2 Complete: {len(speaker_turns)} speaker turns after merging")
            logger.debug(f"Merged turns: {speaker_turns[:5]}...")  # Show first 5

            # STEP 3: Build speaker centroids and map to enrollments
            final_turns = speaker_turns
            if enrollments:
                logger.info("STEP 3: Building speaker centroids for enrollment matching...")
                centroids = self._build_speaker_centroids(audio_path, speaker_turns)
                
                if centroids:
                    logger.info(f"  → Built {len(centroids)} speaker centroids")
                    label_map = self._map_centroids_to_enrollments(centroids, enrollments)
                    
                    # Apply mapping to all turns
                    mapped_count = 0
                    for turn in final_turns:
                        original_label = turn["speaker"]
                        turn["speaker"] = label_map.get(original_label, original_label)
                        if turn["speaker"] != original_label:
                            mapped_count += 1
                    
                    logger.info(f"STEP 3 Complete: Mapped {mapped_count}/{len(final_turns)} turns to enrolled speakers")
                    logger.info(f"  Mapping: {label_map}")
                else:
                    logger.warning("No centroids built (all turns too short). Skipping enrollment matching.")
            else:
                logger.info("STEP 3: Skipped (no enrollments provided)")

            # STEP 4: Extract and validate words from transcription
            logger.info("STEP 4: Extracting words from transcription...")
            words = []
            for segment in transcription_result.get('segments', []):
                words.extend(segment.get('words', []))
            
            # Filter valid words with timestamps
            valid_words = [w for w in words if 'start' in w and 'end' in w and 'word' in w]
            logger.info(f"STEP 4 Complete: Extracted {len(valid_words)} valid words (filtered from {len(words)} total)")
            
            if not valid_words:
                logger.error("No valid words with timestamps found in transcription.")
                return []

            # STEP 5: Align words to speaker turns using INTERVAL OVERLAP + nearest fallback
            logger.info("STEP 5: Aligning words to speaker turns (interval overlap + nearest fallback)...")
            unknown_count = 0

            for word in valid_words:
                word_start = word.get("start", 0)
                word_end = word.get("end", 0)
                word_mid = (word_start + word_end) / 2
                best_speaker = "UNKNOWN"
                max_overlap = 0.0
    
                # First pass: try interval overlap
                for turn in final_turns:
                    turn_start = turn["start"] - ALIGN_TOLERANCE_SEC
                    turn_end = turn["end"] + ALIGN_TOLERANCE_SEC
        
                    overlap_start = max(word_start, turn_start)
                    overlap_end = min(word_end, turn_end)
                    overlap = max(0, overlap_end - overlap_start)
        
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = turn["speaker"]
    
                # Second pass: if still UNKNOWN, assign to nearest turn by midpoint
                if best_speaker == "UNKNOWN" and final_turns:
                    nearest_turn = min(final_turns, key=lambda t: min(
                        abs(word_mid - t["start"]),
                        abs(word_mid - t["end"]),
                        abs(word_mid - (t["start"] + t["end"]) / 2)
                    ))
                    best_speaker = nearest_turn["speaker"]
                    logger.debug(f"  Word '{word.get('word')}' at {word_start:.2f}s → {best_speaker} (nearest fallback)")
    
                word["speaker"] = best_speaker
                if best_speaker == "UNKNOWN":
                    unknown_count += 1
                    logger.warning(f"  Word '{word.get('word')}' at {word_start:.2f}s → UNKNOWN (no turns found)")

            logger.info(f"STEP 5 Complete: Aligned {len(valid_words)} words ({unknown_count} marked UNKNOWN)")
            if unknown_count > len(valid_words) * 0.3:
                logger.warning(f"⚠ High UNKNOWN rate: {unknown_count}/{len(valid_words)} ({unknown_count/len(valid_words)*100:.1f}%)")
                logger.warning("  This suggests timeline mismatch between ASR and diarization.")

            # STEP 6: Merge words into final speaker-labeled segments
            logger.info("STEP 6: Merging words into final speaker segments...")
            output_segments = []
            
            # Sort words by start time
            sorted_words = sorted(valid_words, key=lambda w: w.get("start", 0))
            
            current_segment = {
                'speaker': sorted_words[0]['speaker'],
                'text': sorted_words[0]['word'],
                'start': sorted_words[0]['start'],
                'end': sorted_words[0]['end']
            }

            for word in sorted_words[1:]:
                speaker = word.get('speaker', 'UNKNOWN')
                word_text = word.get('word', '')
                word_start = word.get('start', 0)
                word_end = word.get('end', 0)
                gap = word_start - current_segment['end']

                # Merge if same speaker and gap is small
                if speaker == current_segment['speaker'] and gap <= WORD_MERGE_GAP_SEC:
                    current_segment['text'] += word_text
                    current_segment['end'] = word_end
                else:
                    # Save current and start new
                    output_segments.append(current_segment)
                    logger.debug(f"  Segment: [{current_segment['start']:.2f}-{current_segment['end']:.2f}] "
                               f"{current_segment['speaker']}: {current_segment['text'][:50]}...")
                    current_segment = {
                        'speaker': speaker,
                        'text': word_text,
                        'start': word_start,
                        'end': word_end
                    }

            # Append final segment
            if current_segment:
                output_segments.append(current_segment)
                logger.debug(f"  Final segment: [{current_segment['start']:.2f}-{current_segment['end']:.2f}] "
                           f"{current_segment['speaker']}: {current_segment['text'][:50]}...")

            logger.info(f"STEP 6 Complete: Created {len(output_segments)} final segments")
            
            # Summary statistics
            speaker_dist = defaultdict(int)
            for seg in output_segments:
                speaker_dist[seg['speaker']] += 1
            logger.info(f"{'='*80}")
            logger.info(f"✓ Diarization Complete for {os.path.basename(audio_path)}")
            logger.info(f"  Total segments: {len(output_segments)}")
            logger.info(f"  Speaker distribution: {dict(speaker_dist)}")
            logger.info(f"{'='*80}")

            return output_segments

        except Exception as e:
            logger.error(f"Diarization process failed for {audio_path}: {e}", exc_info=True)
            return []

    def merge_diarization_results(self, all_turns):
        """
        Merge overlapping turns from multiple diarization scales.
        Priority: longer segments in case of conflict.
        """
        logger.debug(f"Merging {len(all_turns)} turns from multiple scales...")
        
        # Sort by start time, then by duration (longer first)
        all_turns = sorted(all_turns, key=lambda x: (x["start"], -(x["end"] - x["start"])))
        
        merged_turns = []
        for turn in all_turns:
            if not merged_turns:
                merged_turns.append(turn)
                continue

            last_turn = merged_turns[-1]
            
            # Calculate overlap
            overlap_start = max(last_turn["start"], turn["start"])
            overlap_end = min(last_turn["end"], turn["end"])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            last_duration = last_turn["end"] - last_turn["start"]
            turn_duration = turn["end"] - turn["start"]
            shorter_duration = min(last_duration, turn_duration)
            
            # Significant overlap if >50% of shorter turn
            has_significant_overlap = (overlap_duration > 0 and 
                                      shorter_duration > 0 and 
                                      (overlap_duration / shorter_duration) > 0.5)
            
            if has_significant_overlap:
                if last_turn["speaker"] == turn["speaker"]:
                    # Same speaker: extend
                    last_turn["end"] = max(last_turn["end"], turn["end"])
                    logger.debug(f"  Merged same-speaker turns: {last_turn['start']:.2f}-{last_turn['end']:.2f}")
                else:
                    # Different speakers: keep longer
                    if turn_duration > last_duration:
                        merged_turns[-1] = turn
                        logger.debug(f"  Replaced with longer turn from scale {turn.get('scale')}")
            else:
                merged_turns.append(turn)

        # Final sort
        final_sorted = sorted(merged_turns, key=lambda x: x["start"])
        logger.debug(f"Merge complete: {len(all_turns)} → {len(final_sorted)} turns")
        return final_sorted

    def _build_speaker_centroids(self, audio_path: str, turns: list) -> dict:
        """
        Build centroid embeddings for each pyannote speaker label.
        Only uses turns >= MIN_TURN_SEC_FOR_EMB for stability.
        """
        logger.info("  Building speaker centroids from long turns...")
        
        full_audio = AudioSegment.from_file(audio_path)
        centroids = {}
        grouped = defaultdict(list)
        
        # Group turns by speaker
        for t in turns:
            grouped[t["speaker"]].append(t)
        
        logger.info(f"  Grouped into {len(grouped)} pyannote speaker labels")

        for spk, spk_turns in grouped.items():
            embs = []
            tmp_files = []
            eligible_count = 0
            
            try:
                for t in spk_turns:
                    dur = t["end"] - t["start"]
                    if dur < MIN_TURN_SEC_FOR_EMB:
                        continue
                    
                    eligible_count += 1
                    start_ms = int(t["start"] * 1000)
                    end_ms = int(t["end"] * 1000)
                    seg = full_audio[start_ms:end_ms]
                    
                    if len(seg) <= 0:
                        logger.debug(f"    Skipping empty segment for {spk} at {t['start']:.2f}s")
                        continue
                    
                    # Export to temp file
                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    seg.export(tmp_wav.name, format="wav")
                    tmp_files.append(tmp_wav.name)
                    
                    # Extract embedding
                    emb = self.voice_recognizer.get_embedding(tmp_wav.name)
                    if emb is not None:
                        # L2 normalize
                        emb = emb / (np.linalg.norm(emb) + 1e-8)
                        embs.append(emb)
                    else:
                        logger.warning(f"    Failed to extract embedding for {spk} at {t['start']:.2f}s")
                
                if embs:
                    # Average all embeddings for this speaker
                    centroid = np.mean(np.stack(embs, axis=0), axis=0)
                    # Normalize centroid
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                    centroids[spk] = centroid
                    logger.info(f"    ✓ {spk}: Built centroid from {len(embs)}/{eligible_count} eligible turns "
                              f"(total {len(spk_turns)} turns)")
                else:
                    logger.warning(f"    ✗ {spk}: No valid embeddings from {eligible_count} eligible turns "
                                 f"(all {len(spk_turns)} turns too short or failed)")
                    
            finally:
                # Cleanup temp files
                for f in tmp_files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        logger.debug(f"Failed to remove temp file {f}: {e}")

        logger.info(f"  Built {len(centroids)}/{len(grouped)} speaker centroids")
        return centroids

    def _map_centroids_to_enrollments(self, centroids: dict, enrollments: list) -> dict:
        """
        Map each diarized speaker centroid to enrolled speaker by best match above threshold.
        """
        logger.info("  Mapping centroids to enrolled speakers...")
        
        mapping = {}
        for spk, centroid in centroids.items():
            best_name = None
            best_score = -1.0
            
            for enr in enrollments:
                enr_emb = enr.get('embedding')
                if enr_emb is None:
                    logger.warning(f"    Enrollment for {enr.get('personName')} has no embedding. Skipping.")
                    continue
                
                # Convert to numpy if needed
                if not isinstance(enr_emb, np.ndarray):
                    enr_emb = np.array(enr_emb)
                
                # Normalize enrollment embedding
                enr_emb = enr_emb / (np.linalg.norm(enr_emb) + 1e-8)
                
                # Verify speaker match
                is_same, score = self.voice_recognizer.verify_speaker(
                    centroid, enr_emb, audio_quality_score=1.0
                )
                
                logger.debug(f"    {spk} vs {enr.get('personName')}: score={score:.3f}, match={is_same}")
                
                if is_same and score > best_score:
                    best_score = score
                    best_name = enr.get('personName')
            
            if best_name:
                logger.info(f"    ✓ {spk} → {best_name} (score={best_score:.3f})")
                mapping[spk] = best_name
            else:
                logger.info(f"    ✗ {spk} → no match (keeping generic label)")
                mapping[spk] = spk
        
        return mapping




