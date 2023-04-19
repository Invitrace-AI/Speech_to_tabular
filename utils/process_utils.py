import torch
import torchaudio
import subprocess
import os
from pydub import AudioSegment
import openai

def speech_file_to_array_fn(audio_path,truth = None, resampling_to=16000):
    
    speech_array, sampling_rate = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(sampling_rate, resampling_to)

    speech = resampler(speech_array)[0].numpy()
    sampling_rate = resampling_to
    target_text = truth
    return speech, sampling_rate, target_text

def predict(speech,processor,model):
    inputs = processor(speech, sampling_rate=16000,return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values,).logits #185 features
    globals()['logits'] = logits
    predicted_ids = torch.argmax(logits, dim=-1)

    prediction = processor.batch_decode(predicted_ids)
    #print("Reference:", test_dataset["sentence"][:2])
    return prediction

def convert_mp3_to_wav(mp3_audio):
    with open("samples/sample.mp3", "wb") as f:
        f.write(mp3_audio.getvalue())
    audio_file = AudioSegment.from_file("samples/sample.mp3")
    audio_file.export("samples/sample.wav", "wav")

    os.remove("samples/sample.mp3")
    #subprocess.call(['ffmpeg', '-i', 'samples/example.mp3','samples/record_0.wav'])

def correct_by_gpt(final_prediction):

    path = 'prompt_template/correction'
    txt_file = '/thai_spell_checker.txt'
    with open(path + txt_file, "r") as f:
        role = f.read()
   
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature = 0.01,
            messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": f'correct this without verbose : {final_prediction}'},
                ])
    

    correct_prediction = ""
    for choice in response.choices:
        correct_prediction += choice.message.content
    
    return correct_prediction

def text_to_tabular(transcription,situation):

    path = 'prompt_template/situation'
    txt_file_dct = {
                    'monologue (doctor)' : '/monologue.txt',
                    "dialogue (doctor-patient)" : '/dialogue.txt'
                    }
    txt_file = txt_file_dct[situation]

    with open(path + txt_file, "r") as f:
        role = f.read()

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature = 0.01,
            messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": f'Please extract the following text into table : {transcription}'},
                ])
    

    tabular_data = ""
    for choice in response.choices:
        tabular_data += choice.message.content
    
    return tabular_data

