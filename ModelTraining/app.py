import os

# os.system('spleeter train -p configs/musdb_config.json -d configs/musdb18hq')
from pydub import AudioSegment

m4a_file = 'data/yes.m4a'
wav_filename = 'data/yes.wav'


track = AudioSegment.from_file(m4a_file, format='m4a')
file_handle = track.export(wav_filename, format='wav')
