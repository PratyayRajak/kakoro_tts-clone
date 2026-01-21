"""Fitness scoring for voice cloning using hybrid similarity metrics."""

from typing import Any, Dict

import librosa
import numpy as np
import scipy.stats
import soundfile as sf
from numpy.typing import NDArray
from resemblyzer import preprocess_wav, VoiceEncoder


class FitnessScorer:
    """Scores voice similarity using a hybrid approach combining multiple metrics.

    The scoring system uses three metrics:
    1. Target Similarity - How similar the generated audio sounds to the target voice
    2. Self Similarity - How consistent the voice is across different text inputs
    3. Feature Similarity - How similar the audio features are to prevent quality degradation
    """

    def __init__(self, target_path: str):
        """Initialize the fitness scorer with a target audio file.

        Args:
            target_path: Path to the target audio file (24kHz mono WAV)
        """
        self.encoder = VoiceEncoder()
        self.target_audio, _ = sf.read(target_path, dtype="float32")
        self.target_wav = preprocess_wav(target_path, source_sr=24000)
        self.target_embed = self.encoder.embed_utterance(self.target_wav)
        self.target_features = self.extract_features(self.target_audio)

    def hybrid_similarity(
        self,
        audio: NDArray[np.float32],
        audio2: NDArray[np.float32],
        target_similarity: float
    ) -> Dict[str, Any]:
        """Calculate hybrid similarity score using weighted harmonic mean.

        The harmonic mean allows controlled backsliding across metrics rather than
        requiring uniform improvement, preventing premature stagnation.

        Args:
            audio: First generated audio sample
            audio2: Second generated audio sample (different text, same voice)
            target_similarity: Pre-computed target similarity score

        Returns:
            Dictionary containing score and all similarity metrics
        """
        features = self.extract_features(audio)
        self_sim = self.self_similarity(audio, audio2)
        target_features_penalty = self.target_feature_penalty(features)

        feature_similarity = (100.0 - target_features_penalty) / 100.0
        if feature_similarity < 0.0:
            feature_similarity = 0.01

        # Weighted harmonic mean scoring
        values = [target_similarity, self_sim, feature_similarity]
        weights = [0.48, 0.5, 0.02]
        score = (np.sum(weights) / np.sum(np.array(weights) / np.array(values))) * 100.0

        return {
            "score": score,
            "target_similarity": target_similarity,
            "self_similarity": self_sim,
            "feature_similarity": feature_similarity
        }

    def target_similarity(self, audio: NDArray[np.float32]) -> float:
        """Calculate similarity between generated audio and target voice.

        Uses Resemblyzer speaker embeddings to measure voice similarity.

        Args:
            audio: Generated audio at 24kHz

        Returns:
            Similarity score between 0 and 1
        """
        audio_wav = preprocess_wav(audio, source_sr=24000)
        audio_embed = self.encoder.embed_utterance(audio_wav)
        similarity = np.inner(audio_embed, self.target_embed)
        return float(similarity)

    def target_feature_penalty(self, features: Dict[str, Any]) -> float:
        """Calculate penalty based on audio feature differences.

        Compares extracted features to prevent convergence on acoustically
        similar but perceptually poor results.

        Args:
            features: Dictionary of extracted audio features

        Returns:
            Penalty value (higher = more different from target)
        """
        penalty = 0.0
        for key, value in features.items():
            if self.target_features[key] != 0:
                diff = abs((value - self.target_features[key]) / self.target_features[key])
                penalty += diff
        return penalty

    def self_similarity(
        self,
        audio1: NDArray[np.float32],
        audio2: NDArray[np.float32]
    ) -> float:
        """Calculate self-similarity between two audio samples from same voice.

        This metric ensures the voice remains consistent across different inputs,
        preventing model degradation during optimization.

        Args:
            audio1: First audio sample
            audio2: Second audio sample

        Returns:
            Similarity score between 0 and 1
        """
        audio_wav1 = preprocess_wav(audio1, source_sr=24000)
        audio_embed1 = self.encoder.embed_utterance(audio_wav1)

        audio_wav2 = preprocess_wav(audio2, source_sr=24000)
        audio_embed2 = self.encoder.embed_utterance(audio_wav2)
        return float(np.inner(audio_embed1, audio_embed2))

    def extract_features(
        self,
        audio: NDArray[np.float32],
        sr: int = 24000
    ) -> Dict[str, Any]:
        """Extract comprehensive audio features for comparison.

        Extracts spectral, temporal, and perceptual features including:
        - Energy and zero-crossing rate
        - Spectral centroid, bandwidth, rolloff, contrast, flatness
        - MFCCs and their deltas
        - Chroma features
        - Mel spectrogram statistics
        - Pitch and tempo

        Args:
            audio: Audio data as numpy array
            sr: Sample rate (default: 24000)

        Returns:
            Dictionary of extracted features
        """
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float64)
        features = {}

        # Energy features
        features["rms_energy"] = float(np.sqrt(np.mean(audio**2)))
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

        n_fft = 2048
        hop_length = 512

        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        features["spectral_contrast_mean"] = float(np.mean(contrast))
        features["spectral_contrast_std"] = float(np.std(contrast))

        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
        )
        for i in range(len(mfccs)):
            features[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc{i+1}_std"] = float(np.std(mfccs[i]))

        # MFCC deltas
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(len(mfcc_delta)):
            features[f"mfcc{i+1}_delta_mean"] = float(np.mean(mfcc_delta[i]))
            features[f"mfcc{i+1}_delta_std"] = float(np.std(mfcc_delta[i]))

        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))
        for i in range(len(chroma)):
            features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        features["mel_spec_mean"] = float(np.mean(mel_spec))
        features["mel_spec_std"] = float(np.std(mel_spec))

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=n_fft, hop_length=hop_length
        )[0]
        features["spectral_flatness_mean"] = float(np.mean(flatness))
        features["spectral_flatness_std"] = float(np.std(flatness))

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features["tonnetz_mean"] = float(np.mean(tonnetz))
        features["tonnetz_std"] = float(np.std(tonnetz))

        # Tempo and beat
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        features["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        if len(beat_frames) > 0:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            if len(beat_times) > 1:
                beat_diffs = np.diff(beat_times)
                features["beat_mean"] = float(np.mean(beat_diffs))
                features["beat_std"] = float(np.std(beat_diffs))
            else:
                features["beat_mean"] = 0.0
                features["beat_std"] = 0.0
        else:
            features["beat_mean"] = 0.0
            features["beat_std"] = 0.0

        # Pitch estimation
        pitches, magnitudes = librosa.core.piptrack(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        pitch_values = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:
                pitch_values.append(pitch)

        if pitch_values:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"] = float(np.std(pitch_values))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0

        # Energy per frame
        energy = np.array([
            np.sum(np.abs(audio[i:i+hop_length]))
            for i in range(0, len(audio), hop_length)
        ])
        features["energy_mean"] = float(np.mean(energy))
        features["energy_std"] = float(np.std(energy))

        # Harmonic ratio
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_squared = S**2
        S_mean = np.mean(S_squared, axis=1)
        S_std = np.std(S_squared, axis=1)
        S_ratio = np.divide(S_mean, S_std, out=np.zeros_like(S_mean), where=S_std != 0)
        features["harmonic_ratio"] = float(np.mean(S_ratio))

        # Audio statistics
        features["audio_mean"] = float(np.mean(audio))
        features["audio_std"] = float(np.std(audio))
        features["audio_skew"] = float(scipy.stats.skew(audio))
        features["audio_kurtosis"] = float(scipy.stats.kurtosis(audio))

        return features
