from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
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
    
class ASR_Core():
    def __init__(self, ckpt_path, sr=16000):
        self.model = EncDecHybridRNNTCTCBPEModel.restore_from(ckpt_path)
        self.model.cuda()
        self.model.eval()

        audio_dir = "/exp/audio"
        if not os.path.exists(audio_dir):
            os.mkdir(audio_dir)
            
        self.audio_dir = audio_dir
        self.sr = sr
    
    def run(self, batch):
        processed_batch = self.preprocess(batch)
        output_batch = self.inference(processed_batch)
        output_batch = self.postprocess(output_batch)
        
        return output_batch
    
    def save_audio(self, audio):
        audio_dir = f"{self.audio_dir}/{os.getpid()}"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        
        audio_path = f'{audio_dir}/audio-{uuid.uuid4()}.wav'
        # if isinstance(np.array, audio):
        #     waveform = audio
        
        # elif isinstance(torch.Tensor, audio):
        waveform = audio.numpy()
            
        sf.write(audio_path, waveform, samplerate=self.sr)
        
        return audio_path

    
    @log_execution_time 
    def inference(self, batch):
        with torch.no_grad():
            outputs = self.model.transcribe(batch, batch_size=len(batch))
        
        transcripts = [output.text for output in outputs]
        
        return transcripts
    
    def preprocess(self, batch):
        processed_batch = []
        for audio in batch:
            audio_path = self.save_audio(audio)
            processed_batch.append(audio_path)
        
        return processed_batch
    
    def postprocess(self, batch):
        
        return batch
    

if __name__ == "__main__":
    asr_core = ASR_Core()
    
    audio_batch = [
        "/home/tuyendv/ai-nlp-stt-service/wav/audio-v1.wav",
    ]

    transcripts = asr_core.run(audio_batch)
    print("transcripts: ", transcripts)