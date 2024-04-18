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


class Database:
    def __init__(self):
        self.audios = []
        self.embeddings = None
        self.max_audio_size = 0

    def __len__(self):
        return len(self.embeddings)

    def upload_vector_to_database(self):
        folders = glob(pathname='data/*')

        for folder in folders:
            files = glob(pathname=f'{folder}/*.wav')

            for file in tqdm(files):
                self.audios.append(file)

                features = extractor(file)

                self.max_audio_size = max(features.shape[1], self.max_audio_size)

                if self.embeddings is None:
                    self.embeddings = features
                else:
                    self.embeddings = np.append(
                        self.embeddings, features
                    )

        print(len(self.embeddings))
        for id in range(len(self.embeddings)):
            padding_length = self.max_audio_size - self.embeddings[id].shape[1]

            self.embeddings[id] = self.embeddings[id] + [0] * padding_length

    def search_similarity_audio(self, audio_file, top_k_results=None):
        if top_k_results is None:
            top_k_results = 3

        features = extractor(audio_file)

        if features.shape[1] > self.max_audio_size:
            features.shape[1] = features.shape[1][:self.max_audio_size]
        else:
            padding_length = self.max_audio_size - features.shape[1]

            features = features + [0] * padding_length

        scores = cosine_scores_numpy(features, self.embeddings)

        indexs = np.argsort(scores)[-top_k_results:][::-1]

        results = list(map(self.audios.__getitem__, indexs))

        return results


if __name__ == '__main__':
    database = Database()
    database.upload_vector_to_database()

    file_path = '/home/namdp/csdl_dpt/data/vietnamese/1.wav'

    results = database.search_similarity_audio(file_path)

    # print(results)
