import os
import json
import argparse
from functools import lru_cache
from natsort import natsorted
from tqdm import tqdm
import transformers
import torch

# Tắt log của transformers để model không spam output
transformers.logging.set_verbosity_error()

# ---- Import các hàm cần dùng từ run_vila ----
from llava.eval.run_vila import main as main_video
from llava.eval.run_vila import load_model_once
try:
    # YÊU CẦU: bạn đã thêm hàm này trong run_vila.py theo hướng dẫn trước đó
    from llava.eval.run_vila import main_image
except Exception as _imp_err:
    main_image = None  # sẽ kiểm tra và báo lỗi rõ ràng ở dưới

# ---- Cache loader để tránh load checkpoint nhiều lần ----
@lru_cache(maxsize=None)
def load_model_cached(model_path: str, conv_mode: str, precision: str):
    tokenizer, model, image_processor = load_model_once(model_path, conv_mode)

    # Chuyển precision
    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.to(torch.bfloat16)
    # fp32: giữ nguyên

    # Bật TF32 nếu GPU hỗ trợ
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return tokenizer, model, image_processor

def extract_id_from_filename(file_name: str) -> str:
    """Lấy id theo quy tắc: file_name.split('_')[1].split('.')[0]"""
    try:
        return file_name.split('_')[1].split('.')[0]
    except Exception:
        return os.path.splitext(file_name)[0]

def run_image_mode(image_path: str,
                   model_path: str,
                   conv_mode: str,
                   query: str,
                   precision: str):
    """Suy luận trên 1 ảnh duy nhất."""
    if main_image is None:
        raise ImportError(
            "Không tìm thấy hàm main_image trong llava.eval.run_vila. "
            "Hãy mở run_vila.py và thêm hàm main_image(...) như hướng dẫn trước đó."
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Kiểm tra đuôi ảnh hợp lệ (không bắt buộc nhưng giúp cảnh báo sớm)
    valid_img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not image_path.lower().endswith(valid_img_exts):
        print(f"[!] Cảnh báo: {image_path} không có đuôi ảnh phổ biến {valid_img_exts}")

    tokenizer, model, image_processor = load_model_cached(model_path, conv_mode, precision)

    output_text = main_image(
        model_path=model_path,
        image_file=image_path,
        query=query,
        conv_mode=conv_mode,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor
    )
    # In ra STDOUT để dễ redirect
    print(output_text.strip() if output_text else "")

def run_video_folder_mode(folder_path: str,
                          output_folder: str,
                          model_path: str,
                          conv_mode: str,
                          query: str,
                          precision: str):
    """Quét cả thư mục video và xuất 1 JSON kết quả như logic cũ."""
    # Thu thập file video
    valid_exts = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    file_names = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    file_names = natsorted(file_names)

    if not file_names:
        print(f"No video files found in folder: {folder_path}")
        return

    # Load model (cache)
    tokenizer, model, image_processor = load_model_cached(model_path, conv_mode, precision)

    results = {}

    # Thêm tqdm để hiển thị tiến trình
    for fname in tqdm(file_names, desc=f"Processing {os.path.basename(folder_path)}", unit="video"):
        video_file = os.path.join(folder_path, fname)
        try:
            output_text = main_video(
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

    # Lưu JSON
    os.makedirs(output_folder, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(folder_path))
    json_file_path = os.path.join(output_folder, f"{folder_name}.json")

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"✅ Saved {len(results)} entries to: {json_file_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate descriptions for image or videos using VILA."
    )
    # Mode mới: image/video
    parser.add_argument(
        '--mode', choices=['image', 'video'], default='video',
        help='Choose prediction mode: image or video'
    )
    # Đường dẫn ảnh khi mode=image
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to a single image when mode=image'
    )
    # Tham số cũ dùng cho video-folder mode
    parser.add_argument(
        '--folder_path', type=str, default=None,
        help='Path to the folder that contains video files (required if mode=video)'
    )
    parser.add_argument(
        '--output_path', type=str, default='/kaggle/working/output',
        help='Path to the output folder (video mode only)'
    )
    # Model & config
    parser.add_argument(
        '--model_path', type=str, default='Efficient-Large-Model/VILA1.5-3b',
        help='Path or HF repo id of the model'
    )
    parser.add_argument(
        '--conv_mode', type=str, default='vicuna_v1',
        help='Conversation mode to use'
    )
    parser.add_argument(
        '--query', type=str,
        default='<video>\n Please describe the video in detail!',
        help='Query prompt; dùng "<image>"/"<video>" tuỳ mode'
    )
    parser.add_argument(
        '--precision', choices=['fp16', 'bf16', 'fp32'], default='fp16',
        help='Precision for model weights'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == 'image':
        if not args.image_path:
            raise ValueError("Please provide --image_path when --mode image")
        # Nếu query còn đang mặc định cho video, thay tag cho đúng (không bắt buộc nhưng rõ ràng hơn)
        query = args.query.replace("<video>", "<image>")
        run_image_mode(
            image_path=args.image_path,
            model_path=args.model_path,
            conv_mode=args.conv_mode,
            query=query,
            precision=args.precision
        )
    else:
        # mode = video
        if not args.folder_path:
            raise ValueError("Please provide --folder_path when --mode video")
        # Nếu query còn đang mặc định cho image, thay tag cho đúng (không bắt buộc)
        query = args.query.replace("<image>", "<video>")
        run_video_folder_mode(
            folder_path=args.folder_path,
            output_folder=args.output_path,
            model_path=args.model_path,
            conv_mode=args.conv_mode,
            query=query,
            precision=args.precision
        )

if __name__ == "__main__":
    main()
