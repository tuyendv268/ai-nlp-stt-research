from transformers import Wav2Vec2ProcessorWithLM, PretrainedConfig
from src.models.wav2vec2 import Wav2Vec2ForCTC
import soundfile as sf
import numpy as np
import uuid
import librosa
import torch
import os

from src.utils.main import (
    log_execution_time,
    pad_2d
)

def load_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio
    
def pad_1d(inputs, pad_value=0.0):
    input_lengths = [len(sample) for sample in inputs]
    max_length = max(input_lengths)
    
    for index in range(len(inputs)):
        if inputs[index].shape[0] < max_length:
            padding = pad_value * torch.ones(max_length - inputs[index].shape[0]).float()
            inputs[index] = torch.cat(
                (inputs[index], padding), dim=0
            )
            
    inputs = torch.stack(inputs, dim=0)
    
    return inputs, input_lengths

class ASR_Core():
    def __init__(
        self, 
        processor_path="exp/ckpts/wav2vec2/processor",
        ckpt_path="exp/ckpts/wav2vec2/model_epoch5_step_final.pt",
        config_path="exp/ckpts/wav2vec2/config.json",
        sr=16000
    ):

        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(processor_path)
        config = PretrainedConfig.from_json_file(config_path)
        self.model = Wav2Vec2ForCTC(config=config)
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval().cuda()

        self.sr = sr
    
    def run(self, batch):
        processed_batch = self.preprocess(batch)

        output_batch = self.inference(processed_batch)
        output_batch = self.postprocess(output_batch)
        
        return output_batch
    
    # @log_execution_time 
    def inference(self, batch):
        input_values, _ = pad_1d(batch, pad_value=0.0)

        with torch.no_grad():
            batch = input_values.cuda()
            outputs = self.model(batch)
                        
        transcripts = [
            self.processor.decode(output, beam_width=100).text 
            for output in outputs.logits.cpu().detach().numpy()
        ]
        
        return transcripts
    
    def preprocess(self, batch):
        processed_batch = []
        for audio in batch:
            input_values = self.processor.feature_extractor(
                audio, sampling_rate=self.sr, return_tensors='pt'
            )["input_values"][0]

            processed_batch.append(input_values)
        
        return processed_batch
    
    def postprocess(self, batch):
        
        return batch
    

if __name__ == "__main__":
    asr_core = ASR_Core()
    
    audio_path = "exp/audio/1210/audio-1aa936fa-a21b-4ee0-8e18-f1d6dcbf63a5.wav"
    
    audio, sr = librosa.load(audio_path, sr=None)
    audio = torch.from_numpy(audio)
    audio_batch = [
        audio,
    ]

    transcripts = asr_core.run(audio_batch)
    print("transcripts: ", transcripts)