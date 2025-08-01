from ray import serve
import torch
import time

from src.cores.capu_core import CaPu_Core
from config.system import Config

class CaPuServe:
    def __init__(self, ckpt_path=None):  
        self.capu_core = CaPu_Core()
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
    async def run(self, batch):
        transcripts = self.capu_core.run(batch)

        return transcripts

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