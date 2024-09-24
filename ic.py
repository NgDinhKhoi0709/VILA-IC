import os
import json
from natsort import natsorted
from llava.eval.run_vila_ic import main, load_model_once

base_folder = '/kaggle/input/dcxvxcvxc'
output_folder = '/kaggle/working'
model_path = 'Efficient-Large-Model/VILA1.5-3b'
conv_mode = 'vicuna_v1'
query = '<image>\\n describe this image with details.'

tokenizer, model, image_processor = load_model_once(model_path, conv_mode)

subfolders = natsorted(os.listdir(base_folder))

for subfolder in subfolders:
    subfolder_path = os.path.join(base_folder, subfolder)
    
    if os.path.isdir(subfolder_path):
        subfolder_results = {}
        
        image_files = natsorted(os.listdir(subfolder_path))

        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)

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
                subfolder_results[image_file] = output_text.strip()
            else:
                print(f"Warning: No output for image {image_file}")
        
        json_file_path = os.path.join(output_folder, f'{subfolder}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(subfolder_results, json_file, ensure_ascii=False, indent=4)
