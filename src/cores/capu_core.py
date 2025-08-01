from src.cores.capu.capu_model import CapuBERTModel

from src.utils.main import (
    log_execution_time,
)

class CaPu_Core():
    def __init__(self):
        self.capu_normalizer = CapuBERTModel(
            vocab_path="src/cores/capu/vocabulary",
            model_paths="dragonSwing/vibert-capu",
            split_chunk=True
        )
    
    def run(self, batch):
        processed_batch = self.preprocess(batch)
        output_batch = self.inference(processed_batch)
        output_batch = self.postprocess(output_batch)
        
        return output_batch
    
    
    @log_execution_time 
    def inference(self, batch):
        output_batch = self.capu_normalizer(batch)
        
        return output_batch
    
    def preprocess(self, batch):
        
        return batch
    
    def postprocess(self, batch):
        processed_batch = []
        for sample in batch:
            if isinstance(sample, str):
                text = sample
            elif isinstance(sample, list):
                text = " ".join(sample)
            else:
                raise Exception("Invalid asr outputs")
                
            processed_batch.append(text)
        
        return processed_batch
    

if __name__ == "__main__":
    capu_core = CaPu_Core()
    
    batch = [
        "theo đó thủ tướng dự kiến tiếp bộ trưởng nông nghiệp mỹ tom wilsack",
        "bộ trưởng thương mại mỹ gina raimondo bộ trưởng tài chính janet yellen",
        "gặp gỡ thượng nghị sĩ patrick leahy và một số nghị sĩ mỹ khác"
    ] 
         
    output_batch = capu_core.run(batch)
    print(output_batch)