# External packages
import streamlit as st
from streamlit_option_menu import option_menu
from audio_recorder_streamlit import audio_recorder
import streamlit.components.v1 as components
from scipy.io.wavfile import read as wav_read, write as wav_write
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
import openai
from openai.error import RateLimitError

# Builtin packages
import io 
import pickle 
import subprocess
import re
import os
import sys
import time

# import modules
from utils.process_utils import speech_file_to_array_fn, predict, convert_mp3_to_wav, correct_by_gpt
from utils.process_utils import text_to_tabular, split_long_audio, empty_folder
from utils.authenticate_utils import authenticate
from utils.styles_util import inject_style

# Setup page title
st.set_page_config(page_title="Speech to Tabular App", page_icon=":studio_microphone:", layout="wide")

def initialize_session_state():
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None 
        st.session_state['audio_source'] = None

@st.cache_resource
def online_import_model(model_name):
    # newest
    online_model_path_dct = {
            'model 1' : 'wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm',
            }
    
    online_model_path = online_model_path_dct[model_name]
    processor = Wav2Vec2Processor.from_pretrained(online_model_path) # แปลงให้เป็น embedding (ใช้ convolution)
    model = Wav2Vec2ForCTC.from_pretrained(online_model_path)
    return processor,model

# Setup model
@st.cache_resource
def local_import_model(model_name):
    # newest
    local_model_path_dct = {
            'model 1' : 'models/model1/',
            }
    
    local_model_path = local_model_path_dct[model_name]
    processor = Wav2Vec2Processor.from_pretrained(local_model_path, local_files_only=True) # แปลงให้เป็น embedding (ใช้ convolution)
    model = Wav2Vec2ForCTC.from_pretrained(local_model_path, local_files_only=True)
    return processor,model

selected = option_menu(
        menu_title= None , 
        options=['About This App','Prediction'],
        icons=['house', 'bi bi-speedometer2'], 
        menu_icon="cast",
        default_index = 1,
        orientation="horizontal")

inject_style()
#initialize_session_state()
authenticate()

if selected == "Prediction":
    # ---- Setup sidebar ---- #
    with st.sidebar:
        st.header("Custom setting :gear:")
        st.selectbox("Choose a situation",options = ("monologue (doctor)", "dialogue (doctor-patient)"), key='situation')
        #st.selectbox("Choose a model",options = ("model 2", "model 1"), key='model_name')
        st.session_state['model_name'] = 'model 1'
        st.selectbox("Choose a way to input your sound",options = ("file uploader", "recorder"), key='input_option')
        st.checkbox("Correct the output (Beta)",key='correct_text')
        processor, model = online_import_model(st.session_state['model_name'])
        #processor, model = local_import_model(st.session_state['model_name'])
        st.session_state['processor'] = processor
        st.session_state['model'] = model

    # Setup Input 
    
    # ---- Setup Audio (Record from user) ----- #
    # ab : audio byte
    uploaded_file = None
    audio_source = None
    with st.container():
        st.subheader('Invitrace Assistant Version 1')
        st.subheader("Step 1 : Input your sound")

        if st.session_state['input_option'] == 'file uploader' :
            ab_from_file_uploader = st.file_uploader("Choose a sound file",key="audio",type = ['wav','mp3'])
            if ab_from_file_uploader:
                audio_source = 'uploader'
                uploaded_file = ab_from_file_uploader
                st.audio(uploaded_file, format="audio/wav")

        else :
            st.write("Record from your mic")
            ab_from_record = audio_recorder(text='',icon_size="5x")
            if ab_from_record:
                audio_source = 'recorder'
                uploaded_file = ab_from_record
                st.audio(uploaded_file, format="audio/wav")

    
    # ----Preprocess Audio file ----- #
    with st.container():
        st.write("---")
        if uploaded_file:
            st.subheader("Result :")
            with st.spinner('Processing the audio file...'):
                # Preprocess for recorder (always .wav)
                if audio_source == 'recorder':
                    # Turn audio_bytes into sr and audio_data
                    with open("samples/sample.wav", "wb") as f:
                            f.write(uploaded_file)

                # Preprocess for uploader (.wav or .mpeg)
                elif audio_source == 'uploader':
                    audio_type = uploaded_file.type[6:] # Extract the type of file 
                    
                    if audio_type == 'mpeg':
                        convert_mp3_to_wav(uploaded_file) # Create new .wav file
                    else:
                        with open("samples/sample.wav", "wb") as f:
                            f.write(uploaded_file.getvalue())

            with st.spinner('Splitting video ..'):
                n_audio_file = split_long_audio("samples/sample.wav")

            with st.spinner('Transcribing the audio...'):
                # Transcribe
                processor = st.session_state['processor']
                model = st.session_state['model']
                final_prediction_all = ''
                for i in range(n_audio_file):
                    speech, _, _ = speech_file_to_array_fn(f'temp_folder/sample_{i}.wav')
                    prediction = predict(speech,processor,model)
                    final_prediction = prediction[0].replace(" ", "")
                    final_prediction_all += final_prediction

                # Display the audio 
                # st.text(f"Token Prediction : {prediction}")
                
                # Full prediction
                #final_prediction = prediction[0].replace(" ", "")
                st.text(f"Final Transcription : {final_prediction_all}")

            empty_folder('temp_folder')

            if st.session_state['correct_text']:
                with st.spinner('Correcting the text ..'):
                    try:
                        correct_prediction = correct_by_gpt(final_prediction_all)
                        st.text(f"Corrected Final Transcription : {correct_prediction}")
                    except RateLimitError:
                        st.warning("Not available now, Try again in a few second")

            # 
            situation = st.session_state['situation']
            with st.spinner(f'Extracting the transcription into tables for {situation}'):
                final_table = text_to_tabular(final_prediction, situation)
                st.text('Final table :')
                st.markdown(f"{final_table}")


elif selected == "About This App":
    with st.container():
        st.subheader("By Invitrace")
        st.title("Speech to Tabular Web application")
        st.write(
            """
        Speech to Tabular Web application is a software program that enables users 
        to convert their speech or audio input into structured data in a tabular format. 
        The application uses advanced Natural Language Processing (NLP) algorithms to 
        understand the spoken language and convert it into structured data.
        """
        )
        
        st.write(
            """
            Here are the columns in our generated structured data
            - Chief Complaint
            - Present Illness	
            - Past Histories	
            - Personal and Social History	
            - Review of Systems	
            - Physical Examination	
            - Problem list	
            - Provisional Diagnosis	
            - Definitive Diagnosis	
            - Plan for Management	
            - Follow-up plans	
            - Plan for Education
            """
        )
        
        st.write("""
        Here is few suggestions to make your output be more accurate : 
        - Record on the low noise environment
        - Speak Clearly
        """)