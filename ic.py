import os
import json
from natsort import natsorted
from llava.eval.run_vila import main
from llava.model.builder import load_pretrained_model
import torch
from transformers import AutoConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

base_folder = '/kaggle/input/xcvbxcvxcv'
output_folder = '/kaggle/working/'
model_path = 'Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix'
conv_mode = 'llama_3'
query = '<image>\\n captioning this image for retrieval text using sbert.'
batch_size = 2  # Process two images at a time, one per GPU

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_model_on_gpu(rank):
    config = AutoConfig.from_pretrained(model_path)
    config.use_cache = False  # Disable KV cache to save memory
    
    # Load the model on the specific GPU
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        device_map=f"cuda:{rank}",
        torch_dtype=torch.float16,
        use_gradient_checkpointing=True,
    )
    
    model = DDP(model, device_ids=[rank])
    return tokenizer, model, image_processor

def process_image(rank, world_size, image_path):
    setup(rank, world_size)
    tokenizer, model, image_processor = load_model_on_gpu(rank)
    
    with torch.no_grad():
        output_text = main(
            model_path=None,
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
    
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        image_paths = [os.path.join(subfolder_path, img) for img in batch]
        
        # Process two images in parallel, one on each GPU
        outputs = mp.spawn(
            process_image,
            args=(2, image_paths[0]),
            nprocs=min(2, len(image_paths)),
            join=True
        )
        
        for j, output_text in enumerate(outputs):
            if output_text:
                subfolder_results[f'{i+j}'] = output_text
                print(f"Processed image {i+j+1}/{len(image_files)}: {batch[j]}")
            else:
                print(f"Warning: No output for image {batch[j]}")
    
    json_file_path = os.path.join(output_folder, f'{subfolder}.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(subfolder_results, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Starting processing...")
    
    subfolders = natsorted(os.listdir(base_folder))
    
    for subfolder in subfolders:
        if os.path.isdir(os.path.join(base_folder, subfolder)):
            print(f"Processing subfolder: {subfolder}")
            process_subfolder(subfolder)
            print(f"Finished processing subfolder: {subfolder}")

    print("All processing completed.")
