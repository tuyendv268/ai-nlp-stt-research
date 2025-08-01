from ray import serve
import torch
import time

from src.cores.asr_core import ASR_Core
from config.system import Config
from src.utils.main import (
    log_execution_time,
    unpack_batch,
    pack_batch
)

class AsrServe:
    def __init__(self, ckpt_path):  
        self.asr_core = ASR_Core(ckpt_path)
        print(f'###load asr ckpt from: {ckpt_path}')

    async def clear_torch_cache(self):    
        torch.cuda.empty_cache()
        
    def preprocess(self, batch):
        return batch
    
    def postprocess(self, batch):
        return batch
    
    @serve.batch(
        max_batch_size=Config.ASR.BATCH_SIZE, 
        batch_wait_timeout_s=Config.ASR.BATCH_WAIT_TIME_OUT
    )
    async def run_asr(self, batch):
        """
        batch = [
            {
                "audio": _audio,
                "segments": _segments
            },
        ]
        """
        batch = list(map(self.segment_audio, batch))
        
        unpacked_batch, batch_ids = unpack_batch(batch)
        transcripts = self.asr_core.run(unpacked_batch)
        transcripts = pack_batch(transcripts, batch_ids=batch_ids)
        
        transcripts = list(map(lambda x: " ".join(x), transcripts))

        return transcripts
    
    @serve.batch(
        max_batch_size=Config.ASR.BATCH_SIZE, 
        batch_wait_timeout_s=Config.ASR.BATCH_WAIT_TIME_OUT
    )
    async def run(self, batch):
        """
        batch = [
            _audio,
            ...
        ]
        """
        processed_batch = list(map(lambda sample: sample["audio"], batch))
        transcripts = self.asr_core.run(processed_batch)

        return transcripts

    def segment_audio(self, input_dict):
        audio = input_dict["audio"]
        segments = input_dict["segments"]
        sample_rate = input_dict["sample_rate"]
        
        audio_list = []
        for _index, (_start_time, _end_time) in enumerate(segments):
            _start_time = int(_start_time * sample_rate)
            _end_time = int(_end_time * sample_rate)
            
            _segment = audio[_start_time: _end_time]
            audio_list.append(_segment)
            
        return audio_list
                                
if __name__ == "__main__":
    import time
    app = AsrServe.bind()
    serve.run(app, host='0.0.0.0', port=1235)
    
    try:
        print("Ray server is running")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Shutting down...')