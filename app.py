import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import torchaudio
from pytube import YouTube
from moviepy.editor import *
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

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
#supported format is mp3 and not wav 
def download_audio(url: str):

    yt = YouTube(url)
    # Extract the video_id from the url
    video_id = yt.video_id

    # Get the first available audio stream and download it
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path="tmp/")

    # Convert the downloaded audio file to mp3 format
    audio_path = os.path.join("tmp/", audio_stream.default_filename)
    audio_clip = AudioFileClip(audio_path)
    mp3_path = os.path.join("tmp/", f"{video_id}.mp3")
    audio_clip.write_audiofile(mp3_path)
    
    # Delete the original audio stream
    os.remove(audio_path)
    return mp3_path

#Load the audio in correct format for transcription
def load_audio(path_audio,input_sample_rate=48000,output_sample_rate=16000):
    # Dataset: convert recorded audio to vector
    waveform, sample_rate = torchaudio.load(path_audio)
    waveform_resampled = torchaudio.functional.resample(waveform, orig_freq=input_sample_rate, new_freq=output_sample_rate) #change sample rate to 16000 to match training. 
    sample = waveform_resampled.numpy()[0]
    return sample

#Transcribe audio
def run_inference(path_audio, output_lang, pipe):
    sample = load_audio(path_audio)
    result = pipe(sample)
      # Debug print to see the structure of the result
    print(f"Result: {result}")
    
    # Extract the text from the result
    transcription = result['text'] if 'text' in result else "Transcription failed"
    return transcription
def get_data_ready(transcript_file: str) -> str:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'sk-proj-pGx63Y4PEJ0JycX7KKGYT3BlbkFJTlWxMLoBYidulGEKWLhx'
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    text_splitter = CharacterTextSplitter()
    llm = Ollama(model = "llama2")
    texts = text_splitter.split_text(transcript_file)
    docs = [Document(page_content=t) for t in texts[:3]]
    return docs,llm

def generate_summary(docs,llm):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary.strip()

def generate_answers(documents,llm,question):

    embeddings = OllamaEmbeddings()
    db = Chroma.from_documents(documents[:20],embeddings)

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa.run(question)

    return answer.strip()
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

    # Set page title
    st.set_page_config(page_title="YouTube Video Summarization", page_icon="📜", layout="wide")
    # Set title
    st.title("YouTube Video Summarizer 🎥", anchor=False)
    st.header("Summarize YouTube videos with AI", anchor=False)
    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")
    choice = st.radio("Please choose an option :", ('Generate Summary', 'Generate Answer to a Question'), horizontal=True)
    # Submit button
    if st.button("Submit") and youtube_url:
        audio_file = download_audio(youtube_url)
        print(f"Audio file: {audio_file} ")
        print("Done!")

        transcribe_text = run_inference(audio_file, 'en',pipe)
        print("Transcription over")

        docs,llm = get_data_ready(transcribe_text)
        # st.write(f"Transcribed Text: {transcribe_text}")
        if choice == "Generate Summary":
            with st.spinner('Generating Summary...'):
                summary = generate_summary(docs,llm)

            st.markdown(f"#### 📃 Video Summary:")
            st.success(summary)
            
        elif choice == "Generate Answer to a Question": 
            st.markdown('#### 🤔 Step 3 : Enter your question')
            question = st.text_input("What are you looking for ?", placeholder="What does X mean ? How to do X ?")
            with st.spinner("Generating answer..."):
                answers = generate_answers(docs,llm,question)
            st.markdown(f"#### 🤖 {question}")
            st.success(answers)
        else:
            st.markdown('#### 🤔 Step 3 : Enter your question')
            question = st.text_input("What are you looking for ?", placeholder="Please enter YouTube URL key first", disabled=True)
            st.success(answers)

if __name__ == "__main__":
    main()