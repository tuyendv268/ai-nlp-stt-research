import json

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from src
from src.serves.asr_serve import AsrServe
from src.serves.vad_serve import VADServe
from src.serves.capu_serve import CaPuServe

class Speech_To_Text_App():
    def __init__(self, vad_ckpt_dir, asr_ckpt_path):
        self.asr_serve = AsrServe(ckpt_path=asr_ckpt_path)
        self.vad_serve = VADServe(ckpt_dir=vad_ckpt_dir)
        self.normalizer = InverseNormalizer(lang='vi')
        self.capu_serve = CaPuServe(ckpt_path=None)
    

    async def run_vad(self, audio, sample_rate):
        # vad
        speech_segments = await self.vad_serve.run(audio)
        speech_segments = [speech_segment for speech_segment in speech_segments if speech_segment[1] - speech_segment[0] > 0.01]
        is_exists_speech = True
        if len(speech_segments) == 0:
            is_exists_speech = False
            
        return {
            "is_exists_speech": is_exists_speech
        }    
            
    
    async def run(self, audio, sample_rate, norm=True):
        # vad
        duration = audio.shape[0] / sample_rate
        if duration > 15:
            speech_segments = await self.vad_serve.run(audio)
            if len(speech_segments) == 0:
                return {
                    "transcript": ""
                }    
        else:
            speech_segments = [[0, duration]]
        print(f"speech_segments: {json.dumps(speech_segments, indent=4)}")
        
        # asr
        input_asr_dict = {
            "audio": audio,
            "segments": speech_segments,
            "sample_rate": sample_rate,
        }
        print("######################################")
        transcript = await self.asr_serve.run_asr(input_asr_dict)
        print(f"asr output: {transcript}")
        if len(transcript.strip()) == 0:
            return {
                    "transcript": ""
                }
        if norm == False:
            return {
                "transcript": transcript,
            }
        transcript = self.normalizer.inverse_normalize(transcript, verbose=False)
        print(f"inverse norm output: {transcript}")
        transcript = await self.capu_serve.run(transcript)
        print(f"capu norm output: {transcript}")
        print("######################################")

        return {
            "transcript": transcript,
        }
        
