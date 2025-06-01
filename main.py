import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import ollama
import altair as alt
import matplotlib.pyplot as plt
import librosa.display
import seaborn as sns
import pandas as pd
import mfcc

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

st.set_page_config(
    page_title="AI Music Analyser",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
col = st.columns((2, 3, 2), gap='medium')

with col[0]:
    st.title("🎧 AI Audio Analyzer")
    st.write("Wgraj plik MP3/WAV i przeanalizuj tempo i tonację.")
    uploaded_file = st.file_uploader("Wybierz plik audio", type=["mp3", "wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpFile:
                tmpFile.write(uploaded_file.read())
                tempPath = tmpFile.name
                st.session_state.tempPath = tempPath
                st.session_state.uploaded = True
        if "originalAnalysis" not in st.session_state:
            originalData, originalY, originalSR = analyzeAudio(tempPath)
            st.session_state.originalAnalysis = originalData
            st.session_state.originalAudioData = originalY
            st.session_state.originalSampleRate = originalSR
                
        with st.form("mod_form", clear_on_submit=False):
            st.markdown("### 🎛️ Dodaj modyfikacje barwy")
            
            if "modifications" not in st.session_state:
                st.session_state.modifications = []
            
            toDelete = None 

            for i, mod in enumerate(st.session_state.modifications):
                st.markdown(f"#### Modyfikacja #{i+1}")
                freq = st.slider(f"🎚️ Zakres częstotliwości #{i+1}", 0, 127, mod["freq"], key=f"freq_{i}")
                change = st.slider(f"🎚️ Zmiana barwy #{i+1}", -3.0, 3.0, mod["change"], 0.1, key=f"change_{i}")
                st.session_state.modifications[i] = {"freq": freq, "change": change}
                if st.form_submit_button(f"🗑️ Usuń modyfikację #{i+1}"):
                    toDelete = i

            if toDelete is not None:
                del st.session_state.modifications[toDelete]
                if not st.session_state.modifications:
                    st.session_state.pop("modified_audio", None)

            add = st.form_submit_button("➕ Dodaj nową modyfikację")
            apply = st.form_submit_button("🎧 Zastosuj modyfikacje")

            if add:
                st.session_state.modifications.append({"freq": (30, 40), "change": 1.0})

            if apply and uploaded_file and st.session_state.modifications:
                try:
                    result_path = mfcc.mfccReconstruction(
                        st.session_state.tempPath,
                        st.session_state.modifications
                    )
                    st.session_state.modified_audio = result_path
                    st.success("✅ Zmieniono barwę i zapisano plik!")
                except Exception as e:
                    st.error(f"Błąd przy zmianie barwy: {e}")
            elif apply and not st.session_state.modifications:
                st.warning("Brak modyfikacji – nie zastosowano zmian.")



        if "modified_audio" in st.session_state:
            st.audio(st.session_state.modified_audio)
        else:
            st.audio(st.session_state.tempPath)

        with col[1]:
            st.subheader("Analiza cech dźwięku")

            try:
                data = st.session_state.originalAnalysis
                y = st.session_state.originalAudioData
                sr = st.session_state.originalSampleRate


                colInside = st.columns((2, 2), gap='medium')

                with colInside[0]:
                    st.markdown("### Cechy podstawowe:")
                    for key, value in data.items():
                        if key != "MFCC":
                            st.markdown(f"- **{key}**: {value}")
                    st.markdown("### Spektrogram logarytmiczny")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
                    fig.colorbar(img, ax=ax, format="%+2.0f dB")
                    ax.set_title('Spektrogram (log-skala)')
                    ax.set_xlabel("Czas [s]")
                    ax.set_ylabel("Częstotliwość [Hz]")
                    st.pyplot(fig)

                    st.markdown ('### Amplituda w czasie')
                    fig, ax = plt.subplots(figsize=(10, 4))
                    librosa.display.waveshow(y, sr=sr, ax=ax)
                    ax.set_title('Amplituda')
                    ax.set_xlabel("Czas [s]")
                    ax.set_ylabel("Amplituda")
                    st.pyplot(fig)


                    st.markdown("### Chroma Feature (rozkład nut w czasie)")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    H  = librosa.feature.chroma_stft(y=y, sr=sr)
                    img = librosa.display.specshow(H, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr, ax=ax)
                    fig.colorbar(img, ax=ax)
                    ax.set(title='Chroma Features')
                    st.pyplot(fig)

                with colInside[1]:
                    st.markdown("### MFCC (średnie wartości):")
                    for i, coef in enumerate(data["MFCC"], 1):
                        st.markdown(f"- MFCC {i}: {coef}")
                    
                    st.markdown('### Heatmap MFCC')
                    mfccData = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    fig, ax = plt.subplots()
                    img = librosa.display.specshow(mfccData, x_axis='time', sr=sr, ax=ax)
                    fig.colorbar(img, ax=ax)
                    ax.set(title='MFCC (czas vs współczynniki)')
                    st.pyplot(fig)

                    st.markdown("### Widmo częstotliwości (średnie)")
                    spectrum = np.abs(librosa.stft(y))
                    mean_spectrum = np.mean(spectrum, axis=1)
                    freqs = librosa.fft_frequencies(sr=sr)

                    df_spectrum = pd.DataFrame({"Frequency [Hz]": freqs, "Amplitude": mean_spectrum})
                    fig2 = plt.figure(figsize=(10, 3))
                    sns.lineplot(data=df_spectrum, x="Frequency [Hz]", y="Amplitude")
                    plt.title("Średnie widmo częstotliwości")
                    plt.xlim(0, 8000)
                    st.pyplot(fig2)

                with col[2]:
                    if st.button("Rozpoznaj gatunek na podstawie cech"):
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
                        - Chroma: {data['Chroma']}


                        Na podstawie tych cech, określ jaki to może być gatunek muzyczny.
                        Uzasadnij odpowiedź. Odpowiedz po polsku.
                                """
                        response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
                        st.markdown("### 🎼 Rozpoznany gatunek muzyczny:")
                        st.write(response['message']['content'])
            except Exception as e:
                st.error(f"Błąd podczas analizy: {str(e)}")
            finally:
                if "modified_audio" in st.session_state:
                    os.unlink(st.session_state.tempPath)

