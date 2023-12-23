import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import llm


def img2text(url):
    image_recognizer = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    return image_recognizer(url)[0]['generated_text']

def tex2speech(text):
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

prompt_template = PromptTemplate.from_template("""
    You are a story teller.
    Create a story based on a simple narrative. No longer, than 20 words.
                            
    CONTEXT: {context}
    STORY:
""")
output_parser = StrOutputParser()

story_chain = prompt_template | llm | output_parser
    


def img2speech(url):
    story = story_chain.invoke({
        "context": img2text(url)
    })

    tex2speech(story)
