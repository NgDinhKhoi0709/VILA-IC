import json
from natsort import natsorted
from llava.eval.run_vila_ic import main, load_model_once
import torch
import os

base_folder = '/kaggle/input/cvbxvbxcv'
output_folder = '/kaggle/working/'
model_path = 'Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix'
conv_mode = 'llama_3'
query = '<image>\\n image captioning this image for retrieval using sbert.'

tokenizer, model, image_processor = load_model_once(model_path, conv_mode)
subfolders = natsorted(os.listdir(base_folder))
batch_size = 6 

for subfolder in subfolders[127:]:
    subfolder_path = os.path.join(base_folder, subfolder)
    if os.path.isdir(subfolder_path):
        subfolder_results = {}
        
        image_files = natsorted(os.listdir(subfolder_path))
        image_batches = []
        for i in range(0, len(image_files), batch_size):
            image_batch = image_files[i:i + batch_size]
            image_paths = [os.path.join(subfolder_path, img_file) for img_file in image_batch]
            image_batches.append(image_paths)

        with torch.no_grad():
            for image_paths in image_batches:
                output_texts = []
                
                # Sử dụng autocast để bật chế độ FP16
                with torch.cuda.amp.autocast():
                    for image_path in image_paths:
                        output_text = main(
                            model_path=model_path,
                            image_file=image_path,
                            query=query,
                            conv_mode=conv_mode,
                            tokenizer=tokenizer, 
                            model=model, 
                            image_processor=image_processor
                        )
                        output_texts.append(output_text.strip() if output_text else None)

                for img_file, output_text in zip(image_paths, output_texts):
                    if output_text:
                        subfolder_results[os.path.basename(img_file)] = output_text
                    else:
                        print(f"Warning: No output for image {os.path.basename(img_file)}")

        json_file_path = os.path.join(output_folder, f'{subfolder}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(subfolder_results, json_file, ensure_ascii=False, indent=4)
