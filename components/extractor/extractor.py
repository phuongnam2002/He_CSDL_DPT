import librosa
import numpy as np
from sklearn.preprocessing import minmax_scale


def extract_mfcc(file_path):
    wave, sample_rate = librosa.load(file_path)

    mfcc = librosa.feature.mfcc(y=wave, sr=sample_rate)

    return mfcc


def extract_chroma(file_path):
    wave, sample_rate = librosa.load(file_path)

    chroma_gram = librosa.feature.chroma_stft(
        y=wave,
        sr=sample_rate,
        hop_length=512
    )

    return chroma_gram


def extract_rms(file_path):
    wave, sample_rate = librosa.load(file_path)

    rms_values = librosa.feature.rms(y=wave)

    return rms_values


def extract_spectral_centroid(file_path):
    wave, sample_rate = librosa.load(file_path)

    centroids = librosa.feature.spectral_centroid(y=wave, sr=sample_rate)

    return centroids


def extract_spectral_rolloff(file_path):
    wave, sample_rate = librosa.load(file_path)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=wave, sr=sample_rate)

    return spectral_rolloff


def mean_pooling(x):
    return np.mean(x, axis=0).reshape(1, -1)


def extractor(file_path):
    mfcc = extract_mfcc(file_path)
    chroma = extract_chroma(file_path)
    rms = extract_rms(file_path)
    spectral_centroid = extract_spectral_centroid(file_path)
    rolloff = extract_spectral_rolloff(file_path)

    features = np.concatenate((mfcc, chroma, rms, spectral_centroid, rolloff), axis=0)

    features = minmax_scale(features, axis=1)

    features = mean_pooling(features)

    return features
