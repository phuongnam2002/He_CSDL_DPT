import os
import uuid
import librosa
import numpy as np
from tqdm import tqdm
from glob import glob
from django.db import models
from frontend import settings
from pydub import AudioSegment
from sklearn.preprocessing import minmax_scale


def convert_to_wav(file_path):
    file_name = file_path.split('.')[0]

    sound = AudioSegment.from_mp3(file_path)
    sound.export(file_name + '.wav', format="wav")

    return file_name + '.wav'


def extract_mfcc(file_path):
    wave, sample_rate = librosa.load(file_path)

    mfcc = librosa.feature.mfcc(y=wave, sr=sample_rate)

    return mfcc


def extract_stft(file_path):
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
    stft = extract_stft(file_path)
    rms = extract_rms(file_path)
    spectral_centroid = extract_spectral_centroid(file_path)
    rolloff = extract_spectral_rolloff(file_path)

    features = np.concatenate((mfcc, stft, rms, spectral_centroid, rolloff), axis=0)

    features = minmax_scale(features, axis=1)

    features = mean_pooling(features)

    return features


def cosine_scores_numpy(compr, refer):
    compr_norm = np.linalg.norm(compr, axis=1)
    refer_norm = np.linalg.norm(refer, axis=1)

    dot_product = np.dot(compr, refer.T)
    similarity = dot_product / (compr_norm[:, np.newaxis] * refer_norm)

    return similarity


def padding(x, max_audio_length):
    if x.shape[1] > max_audio_length:
        x = x[:, :max_audio_length]

    padding_length = max_audio_length - x.shape[1]

    padding = np.zeros((1, padding_length))

    x = np.concatenate((x, padding), axis=1)

    return x


class Database:
    def __init__(self, max_audio_length):
        self.audios = []
        self.embeddings = None
        self.max_audio_length = max_audio_length

    def __len__(self):
        return len(self.embeddings)

    def upload_vector_to_database(self):
        folders = glob(pathname='/home/black/csdl_dpt/data/*')

        for folder in folders:
            files = glob(pathname=f'{folder}/*.wav')

            for file in tqdm(files):
                self.audios.append(file)

                features = extractor(file)
                features = padding(features, self.max_audio_length)

                if self.embeddings is None:
                    self.embeddings = features
                else:
                    self.embeddings = np.append(self.embeddings, features, axis=0)

    def search_similarity_audio(self, audio_file, top_k_results=None):
        if top_k_results is None:
            top_k_results = 3

        features = extractor(audio_file)
        features = padding(features, self.max_audio_length)

        scores = cosine_scores_numpy(features, self.embeddings)

        indexs = np.argsort(scores)[0][::-1][:top_k_results]

        results = list(map(self.audios.__getitem__, indexs))

        return results


def user_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    file_path = os.path.join(settings.FILE_UPLOAD_URL, filename)
    return file_path


# Create your models here.
class File(models.Model):
    file = models.FileField(upload_to=user_directory_path, null=True)
