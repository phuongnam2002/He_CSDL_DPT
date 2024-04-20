from pydub import AudioSegment


def convert_to_wav(file_path):
    file_name = file_path.split('.')[0]

    sound = AudioSegment.from_mp3(file_path)
    sound.export(file_name + '.wav', format="wav")

    return file_name + '.wav'
