from tqdm import tqdm
from glob import glob
import soundfile as sf
import pandas as pd
import torchaudio
import librosa
import json
import os

from matplotlib import pyplot as plt
from src.serves.vad_serve import VADServe

def run(audio_filepaths, vad_ckpt_dir, output_audio_dir, output_vad_metadata_path):
    print(f"processing {len(audio_filepaths)} files.")
    src_sr = 8000
    targ_sr = 16000
    resampler = torchaudio.transforms.Resample(src_sr, targ_sr)

    vad_serve = VADServe(ckpt_dir=vad_ckpt_dir)

    # metadata = []
    with open(output_vad_metadata_path, "w") as f:
        for audio_filepath in tqdm(
            audio_filepaths, desc="Process audio", 
            dynamic_ncols=1, postfix=f"pid={os.getpid()}"
        ):
            filename = os.path.basename(audio_filepath).split(".mp3")[0]
            stereo_audio, sr = torchaudio.load(audio_filepath)
            assert sr == src_sr

            # resample
            stereo_audio = resampler(stereo_audio)
            
            # vad
            timestamps = vad_serve.run_vad(stereo_audio)
            
            # process stereo audio
            for index in range(len(stereo_audio)):
                _mono_audio = stereo_audio[index]
                _timestamps = timestamps[index]
                
                # segment audio
                for _segment_index, (_start_time, _end_time) in enumerate(_timestamps):
                    # _output_segment_path = f'{output_audio_dir}/{filename}-{index}-{_segment_index}-{round(_start_time, 2)}-{round(_end_time, 2)}.wav'
                    _output_segment_path = '{output_audio_dir}/{filename}-{index}-{segment_index}-{start_time}-{end_time}.wav'
                    
                    _output_segment_path = _output_segment_path.format(
                        output_audio_dir=output_audio_dir,
                        filename=filename,
                        index=index,
                        segment_index=_segment_index,
                        start_time=round(_start_time, 2),
                        end_time=round(_end_time, 2),
                    )

                    _start_frame = int(targ_sr * _start_time)
                    _end_frame = int(targ_sr * _end_time)
                    _duration = _end_time - _start_time
                    
                    _segment = _mono_audio[_start_frame: _end_frame]
                    sf.write(_output_segment_path, _segment, samplerate=targ_sr)
                    
                    _segment = {
                        "orig_filename": filename,
                        "audio_filepath": _output_segment_path,
                        "duration": _duration
                    }
                    # metadata.append(_segment)
                    _json_obj = json.dumps(_segment, ensure_ascii=False)
                    f.write(_json_obj + "\n")
                    
                    # if len(metadata) % 1000 == 0:
                    #     with open(output_vad_metadata_path, "w") as f:
                    #         for line in metadata:
                    #             json_obj = json.dumps(line, ensure_ascii=False)
                    #             f.write(json_obj + "\n")
                    #     print(f'### save vad metadata to {output_vad_metadata_path}')
        
        print(f'### save vad metadata to {output_vad_metadata_path}')

    # if len(metadata) % 100 == 0:
    #     with open(output_vad_metadata_path, "w") as f:
    #         for line in metadata:
    #             json_obj = json.dumps(line, ensure_ascii=False)
    #             f.write(json_obj + "\n")
    #     print(f'### save vad metadata to {output_vad_metadata_path}')

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

if __name__ == "__main__":
    audio_dir = "/data/audio/data-telesale-f88/"
    output_audio_dir = "/data2/audio/f88-v1"
    vad_metadata_dir = "/data/asr-research/data/s1_vad_metadata"
    
    vad_ckpt_dir = "/data/ai-nlp-stt-service/exp/ckpts/snakers4_silero-vad_master"
    if not os.path.exists(vad_metadata_dir):
        os.mkdir(vad_metadata_dir)

    audio_filepaths = glob(f'{audio_dir}/*/*.mp3')
    
    print(f"number of audio filepaths: {len(audio_filepaths)}")
    
    n_workers = 2
    batch_size = int(len(audio_filepaths) / n_workers) + 1
    batches_audio_filepaths = split_data_to_batch(data=audio_filepaths, batch_size=batch_size)
    output_vad_metadata_path = '{vad_metadata_dir}/vad_metadata-{index}.jsonl'
    
    params = [
        (
            audio_filepaths,
            vad_ckpt_dir,
            output_audio_dir, 
            output_vad_metadata_path.format(
                vad_metadata_dir=vad_metadata_dir, 
                index=index, 
                # process_id=os.getpid()
                ), 
            ) 
        for index, audio_filepaths in enumerate(batches_audio_filepaths)
    ]
    
    import multiprocessing as mp
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(func=run, iterable=params)
