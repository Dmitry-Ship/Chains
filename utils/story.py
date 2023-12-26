import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

def img2text(url):
    image_recognizer = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    return image_recognizer(url)[0]['generated_text']

def tex2speech(text):
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    path = "speech.wav"
    sf.write(path, speech["audio"], samplerate=speech["sampling_rate"])

    return f'wrote speech to file {path}'
