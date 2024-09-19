import os
import json
from natsort import natsorted
from llava.eval.run_vila import main, load_model_once
import torch

# Đường dẫn cơ sở và thư mục đầu ra
base_folder = "/kaggle/input/videosac"
output_folder = '/kaggle/working/'

# Đường dẫn mô hình và cấu hình inference
model_path = 'Efficient-Large-Model/VILA1.5-3b'  # Sử dụng mô hình video
conv_mode = 'vicuna_v1'  # Cấu hình phù hợp với video
query = '<video>\\n Please describe this video.'

batch_size = 1  # Xử lý từng video một lần
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải mô hình
tokenizer, model, video_processor = load_model_once(model_path)
model = torch.nn.DataParallel(model).to(device)

print(f'Model is loaded on device: {next(model.parameters()).device}')
subfolders = natsorted(os.listdir(base_folder))
id = 0

for subfolder in subfolders:
    subfolder_path = os.path.join(base_folder, subfolder)

    if os.path.isdir(subfolder_path):
        subfolder_results = {}
        video_files = natsorted(os.listdir(subfolder_path))

        for video_file in video_files:
            video_path = os.path.join(subfolder_path, video_file)
            with torch.no_grad():
                output_text = main(
                    model_path=model_path,
                    video_file=video_path,
                    query=query,
                    conv_mode=conv_mode,
                    tokenizer=tokenizer,
                    model=model,
                    video_processor=video_processor
                )

            if output_text:
                subfolder_results[f'{id}'] = output_text.strip()
                print(subfolder_results[f'{id}'])
                id += 1
            else:
                print(f"Warning: No output for video {video_file}")

        json_file_path = os.path.join(output_folder, f'{subfolder}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(subfolder_results, json_file, ensure_ascii=False, indent=4)
