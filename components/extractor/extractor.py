import librosa
import numpy as np
from sklearn.preprocessing import minmax_scale

from extract_feature_function import mfcc, chroma_stft, spectral_centroid, spectral_rolloff, rms


def extract_mfcc(file_path):
    wave, sample_rate = librosa.load(file_path)

    mfcc_array = mfcc(y=wave, sr=sample_rate)

    return mfcc_array


def extract_stft(file_path):
    wave, sample_rate = librosa.load(file_path)

    chroma_gram = chroma_stft(y=wave, sr=sample_rate)

    return chroma_gram


def extract_rms(file_path):
    wave, sample_rate = librosa.load(file_path)

    rms_values = rms(y=wave)

    return rms_values


def extract_spectral_centroid(file_path):
    wave, sample_rate = librosa.load(file_path)

    centroids = spectral_centroid(y=wave, sr=sample_rate)

    return centroids


def extract_spectral_rolloff(file_path):
    wave, sample_rate = librosa.load(file_path)

    spectral_rolloff_value = spectral_rolloff(y=wave, sr=sample_rate)

    return spectral_rolloff_value


def mean_pooling(x):
    return np.mean(x, axis=0).reshape(1, -1)


def extractor(file_path):
    mfcc = extract_mfcc(file_path)
    stft = extract_stft(file_path)
    rms = extract_rms(file_path)
    spectral_centroid = extract_spectral_centroid(file_path)
    rolloff = extract_spectral_rolloff(file_path)

    features = np.concatenate((mfcc, stft, rms, spectral_centroid, rolloff), axis=0)

    features = minmax_scale(features, axis=1)

    features = mean_pooling(features)

    return features
