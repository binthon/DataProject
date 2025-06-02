import streamlit as st
import tempfile
import os
import altair as alt
import mfcc
from connect import genreFeatures
from audioFeatures import analyzeAudio
from plots import *
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
                freq = st.slider(f"Zakres częstotliwości #{i+1}", 0, 127, mod["freq"], key=f"freq_{i}")
                change = st.slider(f"Zmiana barwy #{i+1}", -3.0, 3.0, mod["change"], 0.1, key=f"change_{i}")
                st.session_state.modifications[i] = {"freq": freq, "change": change}
                if st.form_submit_button(f"🗑️ Usuń modyfikację #{i+1}"):
                    toDelete = i

            if toDelete is not None:
                del st.session_state.modifications[toDelete]
                if not st.session_state.modifications:
                    st.session_state.pop("modified_audio", None)

            add = st.form_submit_button("➕ Dodaj nową modyfikację")
            apply = st.form_submit_button("Zastosuj modyfikacje")

            if add:
                st.session_state.modifications.append({"freq": (30, 40), "change": 1.0})

            if apply and uploaded_file and st.session_state.modifications:
                try:
                    result_path = mfcc.mfccReconstruction(
                        st.session_state.tempPath,
                        st.session_state.modifications
                    )
                    st.session_state.modified_audio = result_path
                    st.success("Zmieniono barwę i zapisano plik!")
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
                    st.pyplot(plotSpectrogram(y, sr))

                    st.markdown ('### Amplituda w czasie')
                    st.pyplot(plotWaveform(y, sr))

                    st.markdown("### Chroma Feature (rozkład nut w czasie)")
                    st.pyplot(plotChroma(y, sr))

                with colInside[1]:
                    st.markdown("### MFCC (średnie wartości):")
                    for i, coef in enumerate(data["MFCC"], 1):
                        st.markdown(f"- MFCC {i}: {coef}")
                    
                    st.markdown('### Heatmap MFCC')
                    st.pyplot(plotMfccHeatmap(y, sr))

                    st.markdown("### Widmo częstotliwości (średnie)")
                    st.pyplot(plotMeanSpectrum(y, sr))

                with col[2]:
                    if st.button("Rozpoznaj gatunek na podstawie cech"):
                        genre_info = genreFeatures(data)
                        st.markdown("### 🎼 Rozpoznany gatunek muzyczny:")
                        st.write(genre_info)

            except Exception as e:
                st.error(f"Błąd podczas analizy: {str(e)}")
            finally:
                if "modified_audio" in st.session_state:
                    os.unlink(st.session_state.tempPath)

