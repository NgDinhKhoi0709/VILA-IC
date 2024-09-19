import os
import json
from natsort import natsorted
from llava.eval.run_vila import main, load_model_once

base_folder = '/kaggle/working/data'
output_folder = '/kaggle/working/output'
model_path = 'Efficient-Large-Model/VILA1.5-3b'
conv_mode = 'vicuna_v1'
query = "<video>\n Please describe this video."

tokenizer, model, image_processor = load_model_once(model_path, conv_mode)

# Get the list of subfolders and sort them naturally
subfolders = natsorted(os.listdir(base_folder))

# Loop through each sorted subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(base_folder, subfolder)
    
    if os.path.isdir(subfolder_path):
        # Prepare to collect results for this subfolder
        subfolder_results = {}
        
        # Get the list of image files and sort them naturally
        image_files = natsorted(os.listdir(subfolder_path))

        # Loop through each sorted image file in the subfolder
        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)

            # Call the main function from run_vila.py to process the image
            output_text = main(
                model_path=model_path,
                video_file=image_path,
                query=query,
                conv_mode=conv_mode,
                tokenizer=tokenizer, 
                model=model, 
                image_processor=image_processor
            )
            
            if output_text:
                subfolder_results[image_file] = output_text.strip()  # Strip any extra whitespace
            else:
                print(f"Warning: No output for image {image_file}")
        
        # Save the result of the subfolder to a separate JSON file
        json_file_path = os.path.join(output_folder, f'{subfolder}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(subfolder_results, json_file, ensure_ascii=False, indent=4)
