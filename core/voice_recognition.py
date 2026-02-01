# kairo_pipeline/core/voice_recognition.py
import os
import torch
import numpy as np
from speechbrain.dataio.dataio import read_audio
from speechbrain.inference import EncoderClassifier
from config.logger import logger

class VoiceRecognizer:
    def __init__(self):
        try:
            # Load the pre-trained speaker recognition model
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb")
            )
            logger.info("Voice recognition model loaded successfully.") 
        except Exception as e:
            logger.error(f"Failed to load voice recognition model: {e}", exc_info=True)
            raise

    def get_embedding(self, audio_path: str) -> np.ndarray:
        """
        Generates a speaker embedding from an audio file.
        """
        try:
            signal = read_audio(audio_path)
            with torch.no_grad():
                embedding = self.classifier.encode_batch(signal)
                embedding = embedding.squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for {audio_path}: {e}", exc_info=True)
            return None

    def verify_speaker(self, embedding1: np.ndarray, embedding2: np.ndarray, audio_quality_score: float = 1.0, base_threshold: float = 0.51) -> (bool, float):
        """
        Verifies if two embeddings likely represent the same speaker.
        Returns (is_same_speaker, similarity_score).
        """
        try:
            # L2 normalize both embeddings for cosine stability
            norm1 = np.linalg.norm(embedding1) + 1e-8
            norm2 = np.linalg.norm(embedding2) + 1e-8
            embedding1 = embedding1 / norm1
            embedding2 = embedding2 / norm2

            # Cosine similarity
            similarity = np.dot(embedding1, embedding2)
            # Map [-1, 1] to [0, 1]
            similarity_normalized = (similarity + 1.0) / 2.0

            # Adaptive threshold based on audio quality
            adjusted_threshold = base_threshold * audio_quality_score
            is_same = similarity_normalized >= adjusted_threshold

            logger.debug(f"Similarity: {similarity_normalized:.3f}, Threshold: {adjusted_threshold:.3f}, Match: {is_same}")
            return is_same, similarity_normalized

        except Exception as e:
            logger.error(f"Speaker verification failed: {e}", exc_info=True)
            return False, 0.0
