# 🎧 AI Music Analyzer

Interaktywna aplikacja Streamlit do analizy plików audio (MP3/WAV). Narzędzie wykorzystuje bibliotekę `librosa` do ekstrakcji cech dźwięku i model językowy (Ollama + LLM) do klasyfikacji gatunku muzycznego.

---

## 🚀 Funkcje

✅ Analiza cech akustycznych:
- Tempo, tonacja, długość trwania
- RMS, Zero Crossing Rate
- MFCC, Chroma, Spectral Centroid i inne

✅ Wizualizacje:
- Spektrogram logarytmiczny
- Wykres amplitudy
- Chroma heatmap
- Heatmapa MFCC
- Widmo częstotliwości

✅ Modyfikacja barwy dźwięku:
- Przesunięcie MFCC w zadanych zakresach częstotliwości

✅ Rozpoznanie gatunku muzycznego:
- Opis i uzasadnienie gatunku na podstawie cech akustycznych
- Oparte na modelu językowym wybranym w `ollama`

---

## 🛠️ Wymagania

- Python 3.8+
- Ollama

Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

Uruchomienie lokalnego LLM

```bash
ollama run {wybrany LLM}
```

Uruchomienie aplikacji

```bash
python -m streamlit run main.py
```
