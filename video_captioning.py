import os
import json
import argparse
from functools import lru_cache
from natsort import natsorted
from tqdm import tqdm
import transformers

# Tắt log của transformers
transformers.logging.set_verbosity_error()

from llava.eval.run_vila import main, load_model_once

@lru_cache(maxsize=None)
def load_model_cached(model_path: str, conv_mode: str):
    tokenizer, model, image_processor = load_model_once(model_path, conv_mode)
    return tokenizer, model, image_processor

def extract_id_from_filename(file_name: str) -> str:
    try:
        return file_name.split('_')[1].split('.')[0]
    except Exception:
        return os.path.splitext(file_name)[0]

def chunks(lst, n):
    """Chia list thành các chunk kích thước n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ---- Argument parsing ----
parser = argparse.ArgumentParser(description="Process videos in batches.")
parser.add_argument('--folder_path', type=str, required=True, help='Path to folder containing videos')
parser.add_argument('--output_path', type=str, default='/kaggle/working/output', help='Output folder')
parser.add_argument('--model_path', type=str, default='Efficient-Large-Model/VILA1.5-3b', help='Model path')
parser.add_argument('--conv_mode', type=str, default='vicuna_v1', help='Conversation mode')
parser.add_argument('--query', type=str, default='<video>\n Please describe the video in detail!', help='Prompt')
parser.add_argument('--batch_size', type=int, default=1, help='Number of videos to process per batch')
args = parser.parse_args()

folder_path = args.folder_path
output_folder = args.output_path
model_path = args.model_path
conv_mode = args.conv_mode
query = args.query
batch_size = args.batch_size

tokenizer, model, image_processor = load_model_cached(model_path, conv_mode)

valid_exts = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
if not os.path.isdir(folder_path):
    raise NotADirectoryError(f"Folder not found: {folder_path}")

file_names = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
file_names = natsorted(file_names)

if not file_names:
    print(f"No videos found in folder: {folder_path}")
    exit(0)

results = {}
folder_name = os.path.basename(os.path.normpath(folder_path))

for batch_files in tqdm(list(chunks(file_names, batch_size)), desc=f"Processing {folder_name}", unit="batch"):
    for fname in batch_files:
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
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

os.makedirs(output_folder, exist_ok=True)
json_file_path = os.path.join(output_folder, f"{folder_name}.json")
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

print(f"✅ Saved {len(results)} entries to: {json_file_path}")
