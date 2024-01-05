import torch
from transformers import pipeline, AutoProcessor, BarkModel
from datasets import load_dataset
import scipy
import soundfile as sf

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
    processor = AutoProcessor.from_pretrained("suno/bark")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BarkModel.from_pretrained("suno/bark").to(device)
    voice_preset = "v2/en_speaker_9"

    inputs = processor(text, voice_preset=voice_preset)
    print('generating....')
    audio_array = model.generate(**inputs, pad_token_id=processor.tokenizer.pad_token_id)
    print('squeezing....')
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    print('writing file....')
    scipy.io.wavfile.write("speech-small.wav", rate=sample_rate, data=audio_array)


