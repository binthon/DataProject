import ollama

def genreFeatures(features: dict) -> str:
    prompt = f"""
    Mam utwór muzyczny z następującymi cechami akustycznymi:

    - Tempo: {features['Tempo']} BPM
    - Tonacja: {features['Przybliżona tonacja']}
    - Czas trwania: {features['Czas trwania']} s
    - Zero Crossing Rate: {features['Zero Crossing Rate']}
    - RMS energia: {features['Średnia RMS energia']}
    - Spectral Centroid: {features['Spectral Centroid']}
    - Spectral Rolloff: {features['Spectral Rolloff']}
    - Spectral Bandwidth: {features['Spectral Bandwidth']}
    - Spectral Flatness: {features['Spectral Flatness']}
    - MFCC: {", ".join(features['MFCC'])}
    - Chroma: {features['Chroma']}

    Na podstawie tych cech, określ jaki to może być gatunek muzyczny.
    Uzasadnij odpowiedź. Odpowiedz po polsku.
    """

    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
