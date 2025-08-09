import os
import json
import argparse
from functools import lru_cache
from natsort import natsorted
from llava.eval.run_vila import main, load_model_once

# ---- Cache loader to avoid reloading checkpoint repeatedly ----
@lru_cache(maxsize=None)
def load_model_cached(model_path: str, conv_mode: str):
    # load_model_once is called only the first time for each (model_path, conv_mode)
    tokenizer, model, image_processor = load_model_once(model_path, conv_mode)
    return tokenizer, model, image_processor

def extract_id_from_filename(file_name: str) -> str:
    """
    Lấy id theo quy tắc: file_name.split('_')[1].split('.')[0]
    Ví dụ: 'scene_28227.mp4' -> '28227'
    Nếu không khớp, fallback dùng stem (không đuôi).
    """
    try:
        return file_name.split('_')[1].split('.')[0]
    except Exception:
        return os.path.splitext(file_name)[0]

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process all videos in a folder and generate descriptions.")
parser.add_argument('--folder_path', type=str, required=True,
                    help='Path to the folder that contains video files (e.g., /kaggle/input/scenes-2024/batch_1_scenes_2024/L01_V001)')
parser.add_argument('--output_path', type=str, default='/kaggle/working/output', help='Path to the output folder')
parser.add_argument('--model_path', type=str, default='Efficient-Large-Model/VILA1.5-3b', help='Path to the model')
parser.add_argument('--conv_mode', type=str, default='vicuna_v1', help='Conversation mode to use')
parser.add_argument('--query', type=str, default='<video>\n Please describe the video in detail!', help='Query prompt to describe the video')
args = parser.parse_args()

# Get values from the arguments
folder_path = args.folder_path
output_folder = args.output_path
model_path = args.model_path
conv_mode = args.conv_mode
query = args.query

# Load (cached) model; repeated calls won't reload the checkpoint
tokenizer, model, image_processor = load_model_cached(model_path, conv_mode)

# Collect video files in the folder
valid_exts = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
if not os.path.isdir(folder_path):
    raise NotADirectoryError(f"Folder not found: {folder_path}")

file_names = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
file_names = natsorted(file_names)

if not file_names:
    print(f"No video files found in folder: {folder_path}")
    exit(0)

results = {}

for fname in file_names:
    video_file = os.path.join(folder_path, fname)
    try:
        output_text = main(
            model_path=model_path,
            video_file=video_file,
            query=query,
            conv_mode=conv_mode,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor
        )
        if output_text:
            vid = extract_id_from_filename(fname)
            results[vid] = output_text.strip()
        else:
            print(f"Warning: No output for video {video_file}")
    except Exception as e:
        print(f"Error processing {video_file}: {e}")

# Save all results into a single JSON named after the folder
os.makedirs(output_folder, exist_ok=True)
folder_name = os.path.basename(os.path.normpath(folder_path))
json_file_path = os.path.join(output_folder, f"{folder_name}.json")

# Format: { "id": "output_text", ... }
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

print(f"Saved {len(results)} entries to: {json_file_path}")
