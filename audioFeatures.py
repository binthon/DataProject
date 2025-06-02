import librosa
import pandas as pd
import numpy as np

KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def analyzeAudio(file_path):
    y, sr = librosa.load(file_path)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_chroma = chroma.mean(axis=1)
    top_notes = np.argsort(mean_chroma)[::-1][:3]
    note_names = [KEYS[i] for i in top_notes]

    features = {
        "Tempo": "%.2f" % librosa.beat.beat_track(y=y, sr=sr)[0],
        "Przybliżona tonacja": KEYS[np.argmax(mean_chroma)],
        "Czas trwania": "%.2f" % librosa.get_duration(y=y, sr=sr),
        "Zero Crossing Rate": "%.2f" % librosa.feature.zero_crossing_rate(y).mean(),
        "Średnia RMS energia": "%.2f" % librosa.feature.rms(y=y).mean(),
        "Spectral Centroid": "%.2f" % librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        "Spectral Rolloff": "%.2f" % librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        "Spectral Bandwidth": "%.2f" % librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        "Spectral Flatness": "%.2f" % librosa.feature.spectral_flatness(y=y).mean(),
        "MFCC": ["%.2f" % x for x in mfccs.mean(axis=1)],
        "Chroma": ", ".join(note_names)
    }

    return features, y, sr
