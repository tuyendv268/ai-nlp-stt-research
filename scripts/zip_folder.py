from tqdm import tqdm 
from glob import glob
import tarfile
import json
import os

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

def compress_to_tar_gz(filepaths, output_filepath):
    with tarfile.open(output_filepath, "w:gz") as tar:
        for file in tqdm(filepaths, desc="Tar files", postfix=f'pid={os.getpid()}'):
            tar.add(file, arcname=os.path.basename(file))
    print(f'### zip files to {output_filepath}')

def extract_tar_gz(tar_gz_file, extract_path):
    with tarfile.open(tar_gz_file, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"### extract {tar_gz_file} to {extract_path}")

if __name__ == "__main__":
    type = "zip"
    if type == "zip":
        # zip data
        data_dir = "/data2/audio/f88-v1"
        # filepaths = glob(f'{data_dir}/*.wav')[0:2500000]
        output_dir = "/data/asr-research/zipped"
        
        metadata_path = "/data/asr-research/missing_audio_metadata.jsonl"
        filepaths = list(map(json.loads, open(metadata_path).readlines()))
        filepaths = [sample["audio_filepath"] for sample in filepaths]
        
        filepaths = sorted(filepaths)
        filepaths = filepaths
        n_splits = 16
        n_samples_per_split = int(len(filepaths)/ n_splits) + 1
        
        splits = split_data_to_batch(filepaths, batch_size=n_samples_per_split)
        output_filepath = '{output_dir}/part-{index}.tar.gz'
        params = [
            (param, output_filepath.format(output_dir=output_dir, index=index)) 
            for index, param in enumerate(splits)
        ]

        import multiprocessing as mp
        with mp.Pool(processes=n_splits) as pool:
            pool.starmap(func=compress_to_tar_gz, iterable=params) 
    else:
        # unzip data
        data_dir = "/data/asr-research/zipped"
        filepaths = glob(f'{data_dir}/*.tar.gz')
        output_dir = "/data/asr-research/tmp"
        
        n_splits = len(filepaths)
        params = [
            (param, output_dir) 
            for index, param in enumerate(filepaths)
        ]

        import multiprocessing as mp
        with mp.Pool(processes=n_splits) as pool:
            pool.starmap(func=extract_tar_gz, iterable=params) 