import streamlit as st
from audio_recorder_streamlit import audio_recorder
# import openai
# import base64
import requests
import os
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
import torch
from IPython.display import Audio as display_Audio, display
import torchaudio
# import whisper

#initialize huggingface_api key
# def setup_openai_client(api_key):
os.environ['HUGGINGFACEHUB_API_TOKEN']='YOUR-HUGGINGFACE_API-KEY'
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # return openai.OpenAI(api_key=api_key)

#transcribe audio to text



# Initialize Hugging Face API key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'YOUR_HUGGINGFACE_API_KEY'
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#English transcription full example from https://huggingface.co/openai/whisper-large-v3
device = torch.device('cpu')
torch_dtype = torch.float32

model_id = "openai/whisper-tiny"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)


def load_recorded_audio(path_audio,input_sample_rate=48000,output_sample_rate=16000):
    # Dataset: convert recorded audio to vector
    waveform, sample_rate = torchaudio.load(path_audio)
    waveform_resampled = torchaudio.functional.resample(waveform, orig_freq=input_sample_rate, new_freq=output_sample_rate) #change sample rate to 16000 to match training. 
    sample = waveform_resampled.numpy()[0]
    return sample

def run_inference(path_audio, output_lang, pipe):
    sample = load_recorded_audio(path_audio)
    result = pipe(sample)
      # Debug print to see the structure of the result
    print(f"Result: {result}")
    
    # Extract the text from the result
    transcription = result['text'] if 'text' in result else "Transcription failed"
    return transcription

# def fetch_Ai_response(client,input_text):
#     messages = [{"role":"user","content":input_text}]
#     response = client.chat.completions.create(model="llama2",messages=messages)
#     return response.choices[0].message.content
# def text2speech(message):

#     API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
#     headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

#     payloads = {
#         "inputs": message
#     }
    
#     response = requests.post(API_URL, headers=headers, json=payloads)

#     with open('audio.mp3', 'wb') as file:
#         file.write(response.content)
def main():

    st.sidebar.title("API config")
    api_key = st.sidebar.text_input("enter you openai key", type="password")

    st.title("Speak Easy")
    st.write("Hi there! Click on the voice recorder to interact with me.How can I help you today?")
    recorded_audio = audio_recorder()

    if recorded_audio:
        audio_file = "audio.mp3"
        with open(audio_file,"wb") as f:
            f.write(recorded_audio)
        # Show a spinner while transcribing
        with st.spinner('Transcribing audio...'):
            transcribe_text = run_inference(audio_file, 'en',pipe)
        
        st.write(f"Transcribed Text: {transcribe_text}")
        

if __name__ == "__main__":
    main()