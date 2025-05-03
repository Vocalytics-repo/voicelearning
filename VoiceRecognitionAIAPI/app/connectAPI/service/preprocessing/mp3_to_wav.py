import io
import os
from pydub import AudioSegment

async def mp3_to_wav(file,filename):
    _, file_extension = os.path.splitext(filename.lower())
    if file_extension == ".mp3":
        mp3 = file
        mp3_buffer = io.BytesIO(mp3)
        audio = AudioSegment.from_file(mp3_buffer, format = "mp3")
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format='wav')
        wav_buffer.seek(0)
        return wav_buffer.read()
    else:
        return file