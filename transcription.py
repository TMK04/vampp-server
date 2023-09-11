
import os
import sys
import torch
import librosa
import soundfile as sf
import whisper

def split_audio(audio_file_path,filename, output_directory = "split_audio"):

    audio, sr = librosa.load(audio_file_path, sr=None)
    # Calculate window size
    window_duration = 10  # in seconds
    window_size = int(window_duration * sr)

    # Split the audio
    windows = []
    for i in range(0, len(audio), window_size):
        window = audio[i:i + window_size]
        if len(window) == window_size:
            windows.append(window)
            
    for i, window in enumerate(windows):
        # Remove ".wav" from the filename
        filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_path = os.path.join(output_directory,f"window_{filename}_{i + 1}.mp3")
        sf.write(output_path, window, sr)

def transcribe_and_correct(audio_file_path):
    model = whisper.load_model("base.en")
    result = model.transcribe(audio_file_path)

    

    return result['text']