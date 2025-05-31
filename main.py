import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import ollama

st.set_page_config(page_title="🎵 AI Audio Analyzer", layout="centered")
st.title("🎧 AI Audio Analyzer")
st.write("Wgraj plik MP3/WAV, a my przeanalizujemy tempo i tonację.")

KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def analyzeAudio(file_path):
    y, sr = librosa.load(file_path)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return {
        "Tempo": "%.2f" % librosa.beat.beat_track(y=y, sr=sr)[0],
        "Przybliżona tonacja": KEYS[np.argmax(chroma)],
        "Czas trwania": "%.2f" % librosa.get_duration(y=y, sr=sr),
        "Zero Crossing Rate": "%.2f" % librosa.feature.zero_crossing_rate(y).mean(),
        "Średnia RMS energia": "%.2f" % librosa.feature.rms(y=y).mean(),
        "Spectral Centroid": "%.2f" % librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        "Spectral Rolloff": "%.2f" % librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        "Spectral Bandwidth": "%.2f" % librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        "Spectral Flatness": "%.2f" % librosa.feature.spectral_flatness(y=y).mean(),
        "MFCC": ["%.2f" % x for x in mfccs.mean(axis=1)]
    }

uploaded_file = st.file_uploader("Wybierz plik audio", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.audio(temp_path)

    st.subheader("📊 Analiza cech dźwięku")
    try:
        data = analyzeAudio(temp_path)

        for key, value in data.items():
            if key == "MFCC":
                st.markdown("### 🔠 MFCC (średnie wartości):")
                for i, coef in enumerate(value, 1):
                    st.markdown(f"- MFCC {i}: {coef}")
            else:
                st.markdown(f"- **{key}**: {value}")

    except Exception as e:
        st.error(f"Błąd podczas analizy: {str(e)}")

    if st.button("🧠 Rozpoznaj gatunek na podstawie cech"):
        prompt = f"""
Mam utwór muzyczny z następującymi cechami akustycznymi:

- Tempo: {data['Tempo']} BPM
- Tonacja: {data['Przybliżona tonacja']}
- Czas trwania: {data['Czas trwania']} s
- Zero Crossing Rate: {data['Zero Crossing Rate']}
- RMS energia: {data['Średnia RMS energia']}
- Spectral Centroid: {data['Spectral Centroid']}
- Spectral Rolloff: {data['Spectral Rolloff']}
- Spectral Bandwidth: {data['Spectral Bandwidth']}
- Spectral Flatness: {data['Spectral Flatness']}
- MFCC: {", ".join(data['MFCC'])}

Na podstawie tych cech, określ jaki to może być gatunek muzyczny (np. pop, rock, techno, jazz, ambient, itp.).
Uzasadnij krótko swoją odpowiedź. Odpowiedz po polsku.
        """

        response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
        st.markdown("### 🎼 Rozpoznany gatunek muzyczny:")
        st.write(response['message']['content'])
    os.unlink(temp_path)
