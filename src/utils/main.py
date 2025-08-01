from functools import wraps
import inspect
import asyncio
import torch
import time
import os

def unpack_batch(batch):
    unpacked_batch, batch_ids = [], []
    for batch_id, sequence in enumerate(batch):
        batch_ids += [batch_id] * len(sequence)
        unpacked_batch += sequence

    assert len(unpacked_batch) == len(batch_ids)
    return unpacked_batch, batch_ids

def pack_batch(batch, batch_ids):
    packed_batch = [[] for i in range(max(batch_ids)+1)]
    for batch_id, element in zip(batch_ids, batch):
        packed_batch[batch_id].append(element)

    return packed_batch

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            return wrapper_async(func, *args, **kwargs)
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"LOG_TIME ({os.path.abspath(inspect.getfile(func))}:{func.__name__}()): Executed in {end_time - start_time:.4f} seconds.")
            
            return result
    
    async def wrapper_async(func, *args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"LOG_TIME ({os.path.abspath(inspect.getfile(func))}:{func.__name__}()): Executed in {end_time - start_time:.4f} seconds.")
        
        return result

    return wrapper

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

def pad_2d(inputs, pad_value=0):
    lengths = [len(sample) for sample in inputs]
    max_length = max(lengths)

    for i in range(len(inputs)):
        if inputs[i].shape[0] < max_length:
            padding = pad_value * torch.ones((max_length-inputs[i].shape[0], inputs[i].shape[1]))
            inputs[i] = torch.cat(
                (inputs[i], padding), dim=0
            )
        else:
            inputs[i] = inputs[i][0:max_length]
    inputs = torch.stack(inputs, dim=0)
    return inputs
