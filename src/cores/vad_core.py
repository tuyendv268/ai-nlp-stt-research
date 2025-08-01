from torch import hub
from tqdm import tqdm
import torchaudio
import librosa
import torch
import time
import math

from src.utils.main import (
    log_execution_time,
)

def segment_audio_with_fix_length(audio, sample_rate, max_duration=4.8):    
    segments, lengths = [], []
    audio_length = audio.shape[0]
    segment_length = int(max_duration * sample_rate)

    for index in range(int(audio_length/segment_length)+1):
        segment = audio[index*segment_length: (index+1)*segment_length]
        segment_length = segment.shape[0]

        if segment_length == 0:
            continue
        
        segments.append(segment)
        lengths.append(segment_length)
        
    return segments, lengths

def merge_frame_label_to_group(frame_label):
    labels, offset = [], 0
    start_frame, end_frame = None, None
    for index in range(frame_label.shape[0]-1):     
        if frame_label[index] and not frame_label[index-1]:
            start_frame = offset

        if start_frame is None:
            offset += 1
            continue      

        if frame_label[index] and not frame_label[index+1]:
            end_frame = offset
            labels.append([start_frame, end_frame])
        offset += 1
        
    if start_frame is not None and frame_label[index+1]:
        end_frame = offset + 1
        labels.append([start_frame, end_frame])

    labels = [label for label in labels if len(label) != 0]
    
    return labels  

def postprocess_segments(segments):
    processed_segments = []
    for index in range(len(segments)):
        _segment = segments[index]
        _start_time = _segment[0][0]
        _end_time = _segment[-1][1]
        
        processed_segments.append([_start_time, _end_time])        
        
    return processed_segments

def pad_speech_to_timestamp(timestamp):
    index = 0
    padded_timestamp = []
    for index in range(timestamp.shape[0]-1):
        is_speech = timestamp[index]
        if index > 0:
            if not is_speech and timestamp[index-1]:
                is_speech = True 

        if not is_speech and timestamp[index+1]:
            is_speech = True 
        padded_timestamp.append(is_speech)

    if timestamp[index]:
        is_speech = True 
        padded_timestamp.append(is_speech)
        
    padded_timestamp = torch.tensor(padded_timestamp)
    
    return padded_timestamp

def pad_1d(inputs, pad_value=0):
    input_lengths = [len(sample) for sample in inputs]
    max_length = max(input_lengths)
    
    for index in range(len(inputs)):
        if inputs[index].shape[0] < max_length:
            padding = pad_value * torch.ones(max_length - inputs[index].shape[0])
            inputs[index] = torch.cat(
                (inputs[index], padding), dim=0
            )
            
    inputs = torch.stack(inputs, dim=0)
    
    return inputs, input_lengths

def split_data_to_batch(data, batch_size):
    n_samples = len(data)
    if n_samples % batch_size == 0:
        n_batches = int(len(data)/batch_size)
    else:
        n_batches = int(len(data)/batch_size) + 1

    batches = []
    for i in range(n_batches):
        batch = data[i*batch_size: (i+1)*batch_size]
        batches.append(batch)

    return batches

class VAD_Core():
    def __init__(self, ckpt_dir, sample_rate=16000):
        self.sr = sample_rate

        self.model, self.utils = hub.load(
            repo_or_dir=ckpt_dir, 
            source="local",
            model="silero_vad", 
            force_reload=False, 
            onnx=False
        )
        self.model.cuda()

        self.fn_get_speech_timestamps, self.fn_save_audio, \
            self.fn_read_audio, self.VADIterator, self.fn_collect_chunks = self.utils
            
        self.min_threshold_speech_duration = 0.1
        self.frame_length = 512
    
    def collate_fn(self, batch):
        audios = [segment["audio"] for segment in batch]
        
        audios, lengths = pad_1d(audios, pad_value=0.0)
        frame_lengths = [math.floor(length/self.frame_length) for length in lengths]

        return {
            "audios": audios,
            "lengths": lengths,
            "frame_lengths": frame_lengths
        }

    def preprocess(self, batch):
        processed_batch = []
        for segment in batch:
            start = segment["start"]
            end = segment["end"]
            sample_rate = segment["sample_rate"]
            audio = segment["audio"]
            
            segment = {
                "start": start,
                "end": end,
                "audio": audio,
                "sample_rate": sample_rate
            }

            processed_batch.append(segment)
        
        return processed_batch
    
    def postprocess(self, batch):
        processed_batch = []
        for segments in batch:
            processed_segments = []
            for start_time, end_time in segments:
                if end_time - start_time < self.min_threshold_speech_duration:
                    continue
                
                pred = {
                    "start_time": start_time,
                    "end_time": end_time,
                }
                
                processed_segments.append(pred)
                
            processed_batch.append(processed_segments)
        
        return processed_batch
    
    def run(self, batch, batch_size=None):
        if batch_size is None:
            batch_size = len(batch)
            
        processed_batch = self.preprocess(batch)
        
        output_batch = []
        for _processed_batch in split_data_to_batch(data=processed_batch, batch_size=batch_size):
            _output_batch = self.inference(_processed_batch)
            output_batch += _output_batch
            
        output_batch = self.postprocess(output_batch)
        
        return output_batch
    
    # @log_execution_time
    def inference(self, batch, threshold=0.5):
        padded_batch = self.collate_fn(batch)
        
        audios = padded_batch["audios"]
        frame_lengths = padded_batch["frame_lengths"]

        with torch.no_grad():
            predicts = self.model.audio_forward(audios.cuda(), sr=self.sr)
            vad_results = predicts > threshold

        vad_results = [
            vad_result[0:frame_length] 
            for vad_result, frame_length in zip(vad_results, frame_lengths)
            if frame_length > 1
        ]
        vad_results = list(map(merge_frame_label_to_group, vad_results))
        vad_results = list(
            map(
                lambda x: (torch.tensor(x) * self.frame_length / self.sr).tolist(), 
                vad_results
            )
        )
        
        return vad_results
    
    
if __name__ == "__main__":
    ckpt_dir = "/home/tuyendv/ai-nlp-stt-service/exp/ckpts/snakers4_silero-vad_master"
    vad_core = VAD_Core(ckpt_dir)