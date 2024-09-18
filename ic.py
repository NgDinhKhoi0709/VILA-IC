import os
import json
from natsort import natsorted
from llava.eval.run_vila import main, load_model_once
import torch

base_folder = '/kaggle/input/xcvbxcvxcv'
output_folder = '/kaggle/working/'
model_path = 'Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix'
conv_mode = 'llama_3'
query = '<image>\\n captioning this image for retrieval text using sbert.'
batch_size = 32

# Load the model and wrap with DataParallel for multi-GPU use
tokenizer, model, image_processor = load_model_once(model_path)
model = torch.nn.DataParallel(model).to(device)
subfolders = natsorted(os.listdir(base_folder))
id = 0

for subfolder in subfolders:
    subfolder_path = os.path.join(base_folder, subfolder)
    
    if os.path.isdir(subfolder_path):
        subfolder_results = {}
        image_files = natsorted(os.listdir(subfolder_path))
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]  # Process images in batches         
            for image_file in batch:
                image_path = os.path.join(subfolder_path, image_file)
                with torch.no_grad():
                    output_text = main(
                        model_path=model_path,
                        image_file=image_path,
                        query=query,
                        conv_mode=conv_mode,
                        tokenizer=tokenizer, 
                        model=model, 
                        image_processor=image_processor
                    )
                
                if output_text:
                    subfolder_results[f'{id}'] = output_text.strip()  
                    print(subfolder_results[f'{id}'])
                    id += 1
                else:
                    print(f"Warning: No output for image {image_file}")
        
        json_file_path = os.path.join(output_folder, f'{subfolder}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(subfolder_results, json_file, ensure_ascii=False, indent=4)
