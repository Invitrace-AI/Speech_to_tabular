import streamlit as st
from audio_recorder_streamlit import audio_recorder
import io #input/output
import pickle 
from scipy.io.wavfile import read as wav_read, write as wav_write
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils.process_utils import speech_file_to_array_fn, predict
import re

# Setup model
@st.cache_resource
def import_model():
    with open("model/processor.pkl", "rb") as f:
        processor = pickle.load(f)
        
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
        
    return processor, model

@st.cache_resource
def import_model2():
    processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th") # แปลงให้เป็น embedding (ใช้ convolution)
    model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
    return processor,model

processor, model = import_model2()

# Setup Audio (Record from user)
audio_bytes = audio_recorder()

# If user recorded
if audio_bytes:
    # Turn audio_bytes into sr and audio_data
    wav_io = io.BytesIO(audio_bytes)
    sr, audio_data = wav_read(wav_io)
     
    # Save as record_0.wav
    wav_write('sample/record_0.wav', sr, audio_data)
    
    # Transcribe
    speech, _, _ = speech_file_to_array_fn('sample/record_0.wav')
    prediction = predict(speech,processor,model)
    
    # Display the audio 
    st.audio(audio_bytes, format="audio/wav")
    st.text(f"Prediction : {prediction}")
    
    # Full prediction
    real_prediction = prediction[0].replace(" ", "")
    st.text(f"Real Prediction : {real_prediction}")
