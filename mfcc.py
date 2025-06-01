import torchaudio
import torchaudio.transforms as T
import torch
import soundfile as sf

N_FFT = 2048
HOP = 512

def mfccReconstruction(inputFile, modyfikacja, outputFile="output.wav"):
    waveform, sr = torchaudio.load(inputFile) 

    #1 MelSpectrogram (log)
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        n_mels=128,
        hop_length=HOP,
        mel_scale='htk'
    )
    mel_spec = mel_transform(waveform)  
    log_mel = torch.log(mel_spec + 1e-9) 

    #2 "Oszukana" manipulacja: 
    for mod in modyfikacja:
        r1, r2 = mod["freq"]
        change = mod["change"]
        log_mel[:, r1:r2, :] += change


    #3 log-mel → mel
    mel = torch.exp(log_mel)

    #4 Mel → Spectrogram
    mel_to_spec = T.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=128,
        sample_rate=sr,
    )
    spec = mel_to_spec(mel)

    #5 Griffin-Lim: Spectrogram → Waveform
    griffin = T.GriffinLim(n_fft=N_FFT, hop_length=HOP)
    reconstructed = griffin(spec)

    #6 Zapis
    audio = reconstructed.squeeze()
    if audio.ndim == 2:  
        audio = audio[0]
    sf.write(outputFile, audio.numpy(), sr)
    print(f"Zapisano zmodyfikowany plik jako: {outputFile}")
    return outputFile
