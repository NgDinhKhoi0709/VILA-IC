import os
import json
import torch
from natsort import natsorted
from tqdm import tqdm  # For progress bar

# Define your video processing function
def process_video(video_path, model_path, query, conv_mode, tokenizer, model, image_processor, num_frames):
    # Use the eval_model function to get outputs
    from your_module import eval_model  # Make sure to replace 'your_module' with the actual module name

    # Process video and get the result
    output_text = eval_model(
        model_path=model_path,
        video_file=video_path,
        query=query,
        conv_mode=conv_mode,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        num_video_frames=num_frames
    )
    return output_text

def process_videos_in_subfolders(base_folder, model_path, query, conv_mode, tokenizer, model, image_processor, num_frames=6, batch_size=1):
    # Iterate over each subfolder
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)

        if os.path.isdir(subfolder_path):
            subfolder_results = {}
            video_files = natsorted(os.listdir(subfolder_path))
            id = 1  # Initialize the unique ID for results

            for i in range(0, len(video_files), batch_size):
                batch = video_files[i:i + batch_size]  # Process videos in batches
                
                for video_file in batch:
                    if video_file.endswith(".mp4"):
                        video_path = os.path.join(subfolder_path, video_file)
                        
                        # Process the video
                        with torch.no_grad():
                            output_text = process_video(
                                video_path=video_path,
                                model_path=model_path,
                                query=query,
                                conv_mode=conv_mode,
                                tokenizer=tokenizer,
                                model=model,
                                image_processor=image_processor,
                                num_frames=num_frames
                            )

                        # Save output with unique ID
                        if output_text:
                            subfolder_results[f'{id}'] = output_text.strip()
                            print(f"ID: {id}, Output: {subfolder_results[f'{id}']}")
                            id += 1
                        else:
                            print(f"Warning: No output for video {video_file}")

            # Save results to a JSON file
            output_file = os.path.join(base_folder, f"{subfolder}_results.json")
            with open(output_file, 'w') as f:
                json.dump(subfolder_results, f, indent=4)
            print(f"Results saved to {output_file}")

# Example usage
# Replace with actual model setup and paths
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-folder", type=str, required=True, help="Base folder containing subfolders with video files")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--query", type=str, required=True, help="Query for the model")
    parser.add_argument("--conv-mode", type=str, required=True, help="Conversation mode for the model")
    parser.add_argument("--num-frames", type=int, default=6, help="Number of frames to extract from each video")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing videos")

    args = parser.parse_args()

    # Set up model, tokenizer, and image processor
    from your_module import load_pretrained_model  # Make sure to replace 'your_module' with the actual module name
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path)

    process_videos_in_subfolders(
        base_folder=args.base_folder,
        model_path=args.model_path,
        query=args.query,
        conv_mode=args.conv_mode,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        num_frames=args.num_frames,
        batch_size=args.batch_size
    )
