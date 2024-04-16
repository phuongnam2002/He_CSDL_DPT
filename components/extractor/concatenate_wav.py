import wave
from glob import glob
from tqdm import tqdm
from typing import List


def concatenate_wav(files: List[str], output_name: str):
    data = []
    for infile in files:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()

    output = wave.open(output_name, 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()


cnt = 0

folders = glob(pathname='/home/namdp/csdl_dpt/data/*-F-*')

for folder in tqdm(folders):
    files = glob(pathname=folder + '/*.wav')

    batch_size = 3

    for index in range(0, len(files), batch_size):
        start = index
        end = index + batch_size

        wavs = files[start:end]

        cnt += 1

        output_name = f'/home/namdp/csdl_dpt/data/vietnamese/{cnt}.wav'

        concatenate_wav(wavs, output_name)
