import argparse
import os
import json
from natsort import natsorted
from run_vila import main, load_model_once

def process_images(base_folder, output_folder, model_path, conv_mode, query, batch_size, tokenizer, model, image_processor):
    # Process images
    subfolders = natsorted(os.listdir(base_folder))
    id = 0
    results = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            subfolder_results = {}
            
            image_files = natsorted(os.listdir(subfolder_path))
            
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i + batch_size]
                
                for image_file in batch:
                    image_path = os.path.join(subfolder_path, image_file)
                    
                    output_text = main(
                        model_path=model_path,
                        image_file=image_path,
                        conv_mode=conv_mode,
                        query=query,
                        tokenizer=tokenizer,
                        model=model,
                        image_processor=image_processor
                    )
                    subfolder_results[image_file] = output_text
                    id += 1
            
            results[subfolder] = subfolder_results
    
    # Save results to the output folder
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Process images and extract text using VILA model.')
    parser.add_argument('--base_folder', type=str, required=True, help='Base folder containing images.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output results.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--conv_mode', type=str, default='vqa', help='Conversation mode for processing.')
    parser.add_argument('--query', type=str, default='', help='Query for the model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images.')

    args = parser.parse_args()

    tokenizer, model, image_processor = load_model_once(args.model_path)

    process_images(
        base_folder=args.base_folder,
        output_folder=args.output_folder,
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        query=args.query,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor
    )

if __name__ == "__main__":
    main()
