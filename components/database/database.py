import numpy as np
from tqdm import tqdm
from glob import glob
from components.extractor.extractor import extractor


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
        folders = glob(pathname='data/*')

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

        indexs = np.argsort(scores)[-top_k_results:][::-1]

        results = list(map(self.audios.__getitem__, indexs))

        return results


if __name__ == '__main__':
    database = Database(max_audio_length=400)
    database.upload_vector_to_database()

    audio_file = 'data/vietnamese/1.wav'

    print(database.search_similarity_audio(audio_file))
