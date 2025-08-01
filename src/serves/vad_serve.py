from ray import serve
import torch
import time

from src.cores.vad_core import VAD_Core
from src.utils.main import (
    unpack_batch,
    pack_batch
)

def merge_segment_result(segment_results):
    vad_results = []
    for segment in segment_results:
        sample_rate = segment["sample_rate"]
        start = segment["start"]
        end = segment["end"]
        
        vad_result = segment["vad_result"]
        start_time = start / sample_rate
        for index, segment in enumerate(vad_result):
            for key, value in vad_result[index].items():
                vad_result[index][key] = value + start_time

        vad_results += vad_result
        
    return vad_results

def postprocess_segments(segments):
    processed_segments = []
    for index in range(len(segments)):
        _segment = segments[index]
        _start_time = _segment[0][0]
        _end_time = _segment[-1][1]
        processed_segments.append([_start_time, _end_time])        
        
    return processed_segments

def segment_audio_with_fix_length(audio, sample_rate=16000, max_duration=3.6, min_duration=0.5):    
    audio_length = audio.shape[0]
    max_length = int(round(max_duration * sample_rate, 0))
    min_length = int(round(min_duration * sample_rate, 0))
    
    is_end = False
    segments = []
    for offset in range(0, audio_length, max_length):
        start = offset 
        
        if offset + max_length + min_length >= audio_length:
            is_end = True
            end = audio_length
        else:
            end = offset + max_length
        
        segment = audio[start: end]
        if segment.shape[0] == 0:
            continue
        
        segment = {
            "audio": segment,
            "start": start,
            "end": end,
            "sample_rate": sample_rate,
        }
        segments.append(segment)
        if is_end: break
    
    return segments


def segment_audio_with_speech_timestamps(
    speech_timestamps, padding=0.6,
    short_silence_duration=0.2,
    long_silence_duration=1.5,
    min_segment_duration=1.0,
    max_segment_duration=8.0,
):
    segments, current_segment = [], []
    _prev_end_time, _next_start_time = 0, 0
    for _index in range(len(speech_timestamps)):
        _segment = speech_timestamps[_index]
        
        if _index <= len(speech_timestamps) - 2:
            _next_segment = speech_timestamps[_index+1]
        else:
            _next_segment = None

        _start_time = _segment["start_time"] 
        _end_time = _segment["end_time"]
        
        _next_start_time = _end_time
        if _next_segment is not None:
            _next_start_time = _next_segment["start_time"]
        
        # padding at the end of segment
        _silence_duration = _next_start_time - _end_time
        if _silence_duration > padding:
            _end_time = _end_time + padding
        else:
            _end_time = _end_time + _silence_duration

        # padding in the start of segment
        _silence_duration = _start_time - _prev_end_time
        if _silence_duration > padding:
            _start_time = _start_time - padding
        else:
            _start_time = _prev_end_time
        
        if len(current_segment) == 0: 
            current_segment.append((_start_time, _end_time))
            _prev_end_time = _end_time
            continue
        
        _segment_start_time = current_segment[0][1]
        _segment_end_time = current_segment[-1][-1]
        
        _segment_duration = _segment_end_time - _segment_start_time
        if _segment_duration > max_segment_duration or _silence_duration > long_silence_duration:
            segments.append(current_segment)
            current_segment = []
        elif _silence_duration > short_silence_duration:
            if _segment_duration > min_segment_duration:
                segments.append(current_segment)
                current_segment = []
        
        current_segment.append((_start_time, _end_time))
        _prev_end_time = _end_time
        
    if len(current_segment) != 0:
        segments.append(current_segment)
    
    # postprocess vad result
    processed_segments = []
    for index in range(len(segments)):
        _segment = segments[index]
        _start_time = _segment[0][0]
        _end_time = _segment[-1][1]
        processed_segments.append((_start_time, _end_time))        
    
    # postprocess vad result
    # postprocessed_segments = []
    # _index = 0
    # _is_last = False
    # while _index < len(processed_segments)-1:
    #     _segment = processed_segments[_index]
    #     _next_segment = processed_segments[_index + 1]
        
    #     _start_time = _segment[0] 
    #     _end_time = _segment[1]
        
    #     _next_start_time = _next_segment[0]
    #     _next_end_time = _next_segment[1]
        
    #     _duration = _end_time - _start_time
    #     _next_duration = _next_end_time - _next_start_time
        
    #     if (_next_start_time - _end_time < 0.25) and \
    #         (_next_duration < 6 and _duration < 2) and \
    #         (_next_duration < 2 and _duration < 6):
    #         postprocessed_segments.append((_start_time, _next_end_time))
    #         _index += 2
    #         _is_last = True
    #     else:
    #         postprocessed_segments.append((_start_time, _end_time))
    #         _index += 1
    #         _is_last = False
            
    # if _is_last == False and _index <= len(processed_segments) - 1:
    #     _segment = processed_segments[_index]
        
    #     _start_time = _segment[0] 
    #     _end_time = _segment[1]

    #     postprocessed_segments.append((_start_time, _end_time))
        
    return processed_segments

class VADServe:
    def __init__(self, ckpt_dir):
        self.vad_core = VAD_Core(ckpt_dir=ckpt_dir)
        print(f'###load vad ckpt from: {ckpt_dir}')

    async def clear_torch_cache(self):    
        torch.cuda.empty_cache()
        
    def preprocess(self, batch):
        processed_batch = []
        for sample in batch:
            segments = segment_audio_with_fix_length(
                audio=sample,
                sample_rate=16000, max_duration=4.8, min_duration=0.5
            )
            
            processed_batch.append(segments)
            
        return processed_batch
            
    @serve.batch(
        max_batch_size=64, 
        batch_wait_timeout_s=0.1
    )
    async def run(self, batch):
        print(f"###vad_batch_size: {len(batch)}")
        batch = list(map(segment_audio_with_fix_length, batch))
        unpacked_batch, batch_ids = unpack_batch(batch)
        output_batch = self.vad_core.run(unpacked_batch, batch_size=4*64)

        unpacked_batch = self.postprocess(unpacked_batch, output_batch)
        packed_batch = pack_batch(unpacked_batch, batch_ids=batch_ids)
        packed_batch = list(map(merge_segment_result, packed_batch))
        packed_batch = list(map(segment_audio_with_speech_timestamps, packed_batch))        
        # packed_batch = list(map(postprocess_segments, packed_batch))

        return packed_batch

    def run_vad(self, batch):
        batch = list(map(segment_audio_with_fix_length, batch))
        unpacked_batch, batch_ids = unpack_batch(batch)
        output_batch = self.vad_core.run(unpacked_batch, batch_size=1024)

        unpacked_batch = self.postprocess(unpacked_batch, output_batch)
        packed_batch = pack_batch(unpacked_batch, batch_ids=batch_ids)
        packed_batch = list(map(merge_segment_result, packed_batch))
        packed_batch = list(map(segment_audio_with_speech_timestamps, packed_batch))        
        # packed_batch = list(map(postprocess_segments, packed_batch))

        return packed_batch

    def postprocess(self, batch, output_batch):
        processed_batch = []
        for index in range(len(batch)):
            input_sample = batch[index]
            output_sample = output_batch[index]
            
            audio = input_sample["audio"]
            start = input_sample["start"]
            end = input_sample["end"]
            sample_rate = input_sample["sample_rate"]
            
            vad_result = output_sample
            
            processed_sample = {
                "audio": audio,
                "start": start,
                "end": end,
                "sample_rate": sample_rate,
                "vad_result": vad_result,
            }
            processed_batch.append(processed_sample)
        
        return processed_batch

if __name__ == "__main__":
    import time
    app = VADServe.bind()
    serve.run(app, host='0.0.0.0', port=1235)
    
    try:
        print("Ray server is running")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Shutting down...')