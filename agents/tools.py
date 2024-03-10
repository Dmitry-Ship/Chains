from bs4 import BeautifulSoup
import requests
import torch
from transformers import pipeline, AutoProcessor, BarkModel
from datasets import load_dataset
import scipy
import soundfile as sf
import nltk 
import numpy as np
import time
from langchain.tools import HumanInputRun, Tool, tool
from langchain.chains import LLMMathChain
from infra.llm import llm


def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)


human = HumanInputRun(input_func=get_input)

llm_math_chain = LLMMathChain.from_llm(llm=llm)
calculator =  Tool(
    func=llm_math_chain.invoke,
    name="Calculator",
    description="Useful for when you are asked to perform math calculations"
)

@tool
def fetch_web_page(url: str) -> str:
    """Fetches the web page at the given URL and returns its text content."""
    response = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
    })
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

@tool
def tex2speech_old(text):
    """Uses the Microsoft Speech T5 TTS model to convert text to speech."""
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    path = "speech.wav"
    sf.write(path, speech["audio"], samplerate=speech["sampling_rate"])

    return f'wrote speech to file {path}'

tex2speech_tool_old = Tool(
    func=tex2speech_old,
    name="tex2speech_old",
    description="Uses the Microsoft Speech T5 TTS model to convert text to speech. "
),

@tool
def tex2speech(text):
    """Uses the bark speech model to convert text to speech."""
    voice_preset = "v2/en_speaker_9"
    nltk.download('punkt')

    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small")

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
