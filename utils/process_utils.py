import torch
import torchaudio

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