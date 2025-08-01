from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from fastapi import (
    HTTPException, FastAPI, 
    Body, File, Form
)

from typing import Optional
import soundfile as sf
import librosa
import torch
import time
import json
import io

from src.app import Speech_To_Text_App

import logging
import uuid
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('exp/api.log')
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)

asr_ckpt_path = "/exp/best_model.nemo"      
vad_ckpt_dir = "/exp/ckpts/snakers4_silero-vad_master"
stt_app = Speech_To_Text_App(
    vad_ckpt_dir=vad_ckpt_dir, 
    asr_ckpt_path=asr_ckpt_path
)
app = FastAPI()

@app.post(f"/api/infer")
async def stt_endpoint(
    audio: Optional[bytes] = File(None), 
):
    try:
        audio_dir = "exp/audio/log"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        filename = str(uuid.uuid4())
        audio_filepath = f'{audio_dir}/{filename}.wav'
        audio, sample_rate = sf.read(io.BytesIO(audio))
        sf.write(audio_filepath, audio, samplerate=sample_rate)
    except:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, 
            detail=f"Invalid audio submission. ", headers={}
        )
    try:
        if sample_rate != 16000:
            audio = librosa.resample(
                audio, 
                orig_sr=sample_rate, 
                target_sr=16000
            )

        sample_rate = 16000
        audio = torch.from_numpy(audio).float()

        start_time = time.time()
        response = await stt_app.run(audio, sample_rate, norm=True)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f'###response: {json.dumps(response, indent=4, ensure_ascii=False)}')
        print(f'###execution_time: {execution_time}')
        
        logging.info(f'audio_filepath: {audio_filepath} ----- transcript: {response["transcript"]}')
        
        return response
    
    except Exception as exception:
        message = f"Exception: {str(exception)}"

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=message, headers={}
        )

@app.post(f"/api/stt")
async def stt_endpoint(
    audio: Optional[bytes] = File(None), 
):
    try:
        audio_dir = "exp/audio/log"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        filename = str(uuid.uuid4())
        audio_filepath = f'{audio_dir}/{filename}.wav'
        audio, sample_rate = sf.read(io.BytesIO(audio))
        sf.write(audio_filepath, audio, samplerate=sample_rate)
    except:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, 
            detail=f"Invalid audio submission. ", headers={}
        )
    try:
        if sample_rate != 16000:
            audio = librosa.resample(
                audio, 
                orig_sr=sample_rate, 
                target_sr=16000
            )

        sample_rate = 16000
        audio = torch.from_numpy(audio).float()

        start_time = time.time()
        response = await stt_app.run(audio, sample_rate, norm=False)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f'###response: {json.dumps(response, indent=4, ensure_ascii=False)}')
        print(f'###execution_time: {execution_time}')
        
        logging.info(f'audio_filepath: {audio_filepath} ----- transcript: {response["transcript"]}')
        
        return response
    
    except Exception as exception:
        message = f"Exception: {str(exception)}"

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=message, headers={}
        )

        
@app.post(f"/vad")
async def vad_endpoint(
    audio: Optional[bytes] = File(None), 
):
    try:
        audio, sample_rate = sf.read(io.BytesIO(audio))
    except:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, 
            detail=f"Invalid audio submission. ", headers={}
        )
    try:
        if sample_rate != 16000:
            audio = librosa.resample(
                audio, 
                orig_sr=sample_rate, 
                target_sr=16000
            )

        sample_rate = 16000
        audio = torch.from_numpy(audio).float()

        start_time = time.time()
        response = await stt_app.run_vad(audio, sample_rate)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f'###response: {json.dumps(response, indent=4, ensure_ascii=False)}')
        print(f'###execution_time: {execution_time}')
        
        return response
    
    except Exception as exception:
        message = f"Exception: {str(exception)}"

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=message, headers={}
        )


print(f"###STT server is running")

