import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import seaborn as sns

def plotSpectrogram(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Spektrogram (log-skala)')
    return fig

def plotWaveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Amplituda')
    return fig

def plotChroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title('Chroma Features')
    return fig

def plotMfccHeatmap(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title('MFCC (czas vs współczynniki)')
    return fig

def plotMeanSpectrum(y, sr):
    spectrum = np.abs(librosa.stft(y))
    mean_spectrum = np.mean(spectrum, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    df = pd.DataFrame({"Frequency [Hz]": freqs, "Amplitude": mean_spectrum})
    fig = plt.figure(figsize=(10, 3))
    sns.lineplot(data=df, x="Frequency [Hz]", y="Amplitude")
    plt.title("Średnie widmo częstotliwości")
    plt.xlim(0, 8000)
    return fig
