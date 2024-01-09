import torch
from transformers import pipeline, AutoProcessor, BarkModel
from datasets import load_dataset
import scipy
import soundfile as sf
import nltk 
import numpy as np
import time

def img2text(url):
    image_recognizer = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    return image_recognizer(url)[0]['generated_text']

def tex2speech_old(text):
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    path = "speech.wav"
    sf.write(path, speech["audio"], samplerate=speech["sampling_rate"])

    return f'wrote speech to file {path}'

def tex2speech(text):
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small")

    voice_preset = "v2/en_speaker_9"

    sentences = nltk.sent_tokenize(text.replace("\n", " ").strip())
    silence = np.zeros(int(0.25 * model.generation_config.sample_rate))  # quarter second of silence

    pieces = []
    for i, sentence in enumerate(sentences):
        inputs = processor(sentence, voice_preset=voice_preset)
        print(f'generating audio for sentence {i + 1}/{len(sentences)} ...')
        audio_array = model.generate(**inputs, pad_token_id=processor.tokenizer.pad_token_id)
        pieces += [audio_array.cpu().numpy().squeeze()]

    curr_time = round(time.time()* 1000)
    path = f"speech-{curr_time}.wav"
    print(f'writing file {path}...') 

    scipy.io.wavfile.write(
        path, 
        rate=model.generation_config.sample_rate, 
        data=np.concatenate(pieces)
    )

