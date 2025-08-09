import os
import json
import argparse
from natsort import natsorted
from llava.eval.run_vila import main, load_model_once

# Set up argument parsing for the input video file
parser = argparse.ArgumentParser(description="Process a video file and generate festival descriptions.")
parser.add_argument('--video_path', type=str, default='sample_videos/input.mp4', help='Path to the input video file')
parser.add_argument('--output_path', type=str, default='/kaggle/working/output', help='Path to the output folder')
parser.add_argument('--model_path', type=str, default='Efficient-Large-Model/VILA1.5-3b', help='Path to the model')
parser.add_argument('--conv_mode', type=str, default='vicuna_v1', help='Conversation mode to use')
parser.add_argument('--query', type=str, default='<video>\n Please describe the video in detail!', help='Query prompt to describe the video')
parser.add_argument('--file_name', type=str, default='video.json', help='Name of the output JSON file')

args = parser.parse_args()

# Get values from the arguments
video_path = args.video_path
output_folder = args.output_path
model_path = args.model_path
conv_mode = args.conv_mode
query = args.query
file_name = args.file_name

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
    # Define the JSON file path based on --file_name
    json_file_path = os.path.join(output_folder, file_name)

    # Save the result in a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump({file_name.split('_')[1].split('.')[0]: output_text.strip()}, json_file, ensure_ascii=False, indent=4)
else:
    print(f"Warning: No output for video {video_path}")

