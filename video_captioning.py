import os
import json
import argparse
from natsort import natsorted
from llava.eval.run_vila import main, load_model_once

# Set up argument parsing for the input video file
parser = argparse.ArgumentParser(description="Process a video file and generate festival descriptions.")
parser.add_argument('--video', type=str, required=True, help='Path to the input video file')

args = parser.parse_args()

# Get the path to the video file from arguments
video_path = args.video
output_folder = '/kaggle/working/output'  # Fixed output folder
model_path = 'Efficient-Large-Model/VILA1.5-3b'
conv_mode = 'vicuna_v1'
query = "<video>\n Please describe the video in detail!"

tokenizer, model, image_processor = load_model_once(model_path, conv_mode)

# Process the video file
output_text = main(
    model_path=model_path,
    video_file=video_path,
    query=query,
    conv_mode=conv_mode,
    tokenizer=tokenizer, 
    model=model, 
    image_processor=image_processor
)

# Save the result to a JSON file
if output_text:
    # Define the JSON file path based on the video file name
    video_filename = os.path.basename(video_path)
    json_file_path = os.path.join(output_folder, 'video.json')

    # Save the result in a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump({video_filename: output_text.strip()}, json_file, ensure_ascii=False, indent=4)
else:
    print(f"Warning: No output for video {video_path}")

