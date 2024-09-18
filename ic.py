import os
import json
from natsort import natsorted
from llava.eval.run_vila import main, load_model_once
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

base_folder = '/kaggle/input/xcvbxcvxcv'
output_folder = '/kaggle/working/'
model_path = 'Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix'
conv_mode = 'llama_3'
query = '<image>\\n captioning this image for retrieval text using sbert.'
batch_size = 32

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_model_parallel(rank, world_size):
    setup(rank, world_size)
    
    # Load the model
    tokenizer, model, image_processor = load_model_once(model_path)
    
    # Move model to the appropriate device
    device = torch.device(f'cuda:{rank}')
    model.to(device)
    
    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[rank])
    
    return tokenizer, model, image_processor

def process_image(rank, world_size, image_path, query):
    tokenizer, model, image_processor = load_model_parallel(rank, world_size)
    
    with torch.no_grad():
        output_text = main(
            model_path=None,  # We don't need this as we've already loaded the model
            image_file=image_path,
            query=query,
            conv_mode=conv_mode,
            tokenizer=tokenizer, 
            model=model, 
            image_processor=image_processor
        )
    
    cleanup()
    return output_text.strip() if output_text else None

def process_subfolder(subfolder):
    subfolder_path = os.path.join(base_folder, subfolder)
    subfolder_results = {}
    image_files = natsorted(os.listdir(subfolder_path))
    id = 0
    
    for image_file in image_files:
        image_path = os.path.join(subfolder_path, image_file)
        
        # Use multiprocessing to run the model across GPUs
        output_text = mp.spawn(
            process_image,
            args=(2, image_path, query),
            nprocs=2,
            join=True
        )[0]  # Get the result from the first process
        
        if output_text:
            subfolder_results[f'{id}'] = output_text
            print(subfolder_results[f'{id}'])
            id += 1
        else:
            print(f"Warning: No output for image {image_file}")
    
    json_file_path = os.path.join(output_folder, f'{subfolder}.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(subfolder_results, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    subfolders = natsorted(os.listdir(base_folder))
    
    for subfolder in subfolders:
        if os.path.isdir(os.path.join(base_folder, subfolder)):
            process_subfolder(subfolder)
