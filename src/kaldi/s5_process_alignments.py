from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
import librosa
import json
import re

def convert_phone_to_word_alignment(script, alignment):
    words = script.split()
    word_alignment = [None] * len(words)

    _current_word_index = -1
    _arpabet, _start_time, _end_time = [], 0, 0
    for _phone, _start, _end in alignment:
        if _phone.endswith("B"):
            _current_word_index += 1
            _start_time = _start
        elif _phone.endswith("S"):
            _current_word_index += 1
            _start_time = _start
            _end_time = _end
        elif _phone.endswith("E"):
            _end_time = _end

        _arpabet.append(_phone)
        if _phone.endswith("S") or _phone.endswith("E"):
            _arpabet = " ".join(_arpabet)
            word_alignment[_current_word_index] = [
                words[_current_word_index], 
                _start_time, _end_time, 
                _arpabet
            ]
            _arpabet = []
            
    return word_alignment

def process_line(line):
    line = line.split("\n")
    
    frame_sample_id, *frames = line[0].split()
    phoneme_sample_id, *phonemes = line[1].split()

    frames_str = " ".join(frames)
    phonemes_str = " ".join(phonemes)

    frames_list = re.split(r'(?<=\])\s(?=\[)', frames_str)
    frames_list = list(map(lambda x: x.strip().strip("[").strip("]").split(), frames_list))
    phoneme_list = phonemes_str.split()

    assert frame_sample_id == phoneme_sample_id
    
    offset = 0
    aligments = []
    for index in range(len(frames_list)):
        phoneme = phoneme_list[index]
        frame = frames_list[index]
        
        start = offset
        end = offset + len(frame)
        
        offset = end
        if phoneme == "SIL":
            continue
        aligments.append([phoneme, start, end])

    return phoneme_sample_id, aligments

def load_alignment_result(filepath):
    lines = open(filepath).read().split("\n\n")
    
    alignment_dict = {}
    for line in tqdm(lines, desc="Load data"):
        if len(line.strip()) <= 0:
            continue
        
        filename, alignment = process_line(line)
        alignment_dict[filename] = alignment
        
    return alignment_dict

def load_text_file(filepath):
    data_dict = {}
    for line in tqdm(open(filepath).readlines(), desc="Load data"):
        filename, line = line.split("\t")
        data_dict[filename] = line
        
    return data_dict

def process_alignment(data_dir, align_dir, output_filepath):
    wav_scp_path = f'{data_dir}/wav.scp'
    text_path = f'{data_dir}/text'

    audio_path_dict = load_text_file(wav_scp_path)
    text_dict = load_text_file(text_path)
    
    with open(output_filepath, "w") as f:
        for filepath in tqdm(glob(f'{align_dir}/*.txt'), desc="Process"):
            
            alignment_dict = load_alignment_result(filepath=filepath)
            for key in tqdm(alignment_dict.keys(), desc="Process"):
                transcript = text_dict[key].strip()
                audio_path = audio_path_dict[key].strip()
                
                alignments = alignment_dict[key]
                
                word_alignments = convert_phone_to_word_alignment(
                    script=transcript, 
                    alignment=alignments
                    )
                
                alignments = []
                for segment in word_alignments:
                    word = segment[0]
                    start_time = segment[1] * 0.01
                    end_time = segment[2] * 0.01
                    
                    segment = [word, start_time, end_time]
                    alignments.append(segment)
                
                sample = {
                    "id": key,
                    "text": transcript,
                    "audio_filepath": audio_path,
                    "alignments": alignments
                }
                json_obj = json.dumps(sample, ensure_ascii=False)
                f.write(json_obj + "\n")
                # metadata.append(sample)
    print(f'### save data to {output_filepath}')

if __name__ == "__main__":
    data_dir = "/data/asr-research/src/kaldi/data/f88_infer"
    align_dir = "/data/asr-research/src/kaldi/exp/exp_mfcc_pitch/tri5a_f88_infer_out"
    output_filepath = f"{data_dir}/ali.jsonl"
    
    process_alignment(
        data_dir=data_dir, 
        align_dir=align_dir, 
        output_filepath=output_filepath
    )