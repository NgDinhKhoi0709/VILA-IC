import os, json, argparse, multiprocessing as mp
from functools import lru_cache
from natsort import natsorted
from tqdm import tqdm
import transformers, torch

transformers.logging.set_verbosity_error()

from llava.eval.run_vila import main, load_model_once

# ===== Globals cho worker =====
G_FOLDER_PATH = G_MODEL_PATH = G_CONV_MODE = G_QUERY = None
G_TOKENIZER = G_MODEL = G_IMAGE_PROCESSOR = None
PRECISION = "fp16"  # default

@lru_cache(maxsize=None)
def _load_model_cached(model_path: str, conv_mode: str):
    tokenizer, model, image_processor = load_model_once(model_path, conv_mode)
    # Áp dụng precision
    if PRECISION == "fp16":
        model = model.half()
    elif PRECISION == "bf16":
        model = model.to(torch.bfloat16)
    # Bật TF32 (nếu GPU hỗ trợ) để tăng tốc
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tokenizer, model, image_processor

def _init_worker(folder_path, model_path, conv_mode, query, precision):
    """Chạy 1 lần ở mỗi worker."""
    global G_FOLDER_PATH, G_MODEL_PATH, G_CONV_MODE, G_QUERY
    global G_TOKENIZER, G_MODEL, G_IMAGE_PROCESSOR, PRECISION
    G_FOLDER_PATH, G_MODEL_PATH, G_CONV_MODE, G_QUERY = folder_path, model_path, conv_mode, query
    PRECISION = precision
    G_TOKENIZER, G_MODEL, G_IMAGE_PROCESSOR = _load_model_cached(G_MODEL_PATH, G_CONV_MODE)

def extract_id_from_filename(file_name: str) -> str:
    # id = file_name.split('_')[1].split('.')[0]  (fallback: stem)
    try:
        return file_name.split('_')[1].split('.')[0]
    except Exception:
        return os.path.splitext(file_name)[0]

def _process_batch(batch_files):
    """Hàm chạy trong worker: xử lý 1 nhóm video, trả về dict {id: text}."""
    local_results = {}
    for fname in batch_files:
        try:
            video_file = os.path.join(G_FOLDER_PATH, fname)
            out = main(
                model_path=G_MODEL_PATH,
                video_file=video_file,
                query=G_QUERY,
                conv_mode=G_CONV_MODE,
                tokenizer=G_TOKENIZER,
                model=G_MODEL,
                image_processor=G_IMAGE_PROCESSOR
            )
            if out:
                vid = extract_id_from_filename(fname)
                local_results[vid] = out.strip()
        except Exception:
            # Bỏ qua video lỗi, có thể log chi tiết nếu cần
            pass
    # trả về tuple (số video trong batch, dict kết quả) để cập nhật tqdm chính xác
    return (len(batch_files), local_results)

def chunks(lst, n):
    """Chia list thành các nhóm kích thước n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main_entry():
    p = argparse.ArgumentParser(description="Process videos in folder with multiprocessing + batching.")
    p.add_argument('--folder_path', required=True, help='Folder chứa các file video')
    p.add_argument('--output_path', default='/kaggle/working/output', help='Folder lưu JSON')
    p.add_argument('--model_path', default='Efficient-Large-Model/VILA1.5-3b', help='Model path')
    p.add_argument('--conv_mode', default='vicuna_v1', help='Conversation mode')
    p.add_argument('--query', default='<video>\n Please describe the video in detail!', help='Prompt')
    p.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2)//2), help='Số process chạy song song')
    p.add_argument('--precision', choices=['fp16','bf16','fp32'], default='fp16', help='Độ chính xác khi load model')
    p.add_argument('--batch_size', type=int, default=4, help='Số video trong mỗi nhóm giao cho 1 worker')
    args = p.parse_args()

    folder_path = args.folder_path
    output_folder = args.output_path
    batch_size = max(1, args.batch_size)

    exts = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
    files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])
    if not files:
        print(f"No video files found in folder: {folder_path}")
        return

    batches = list(chunks(files, batch_size))

    results = {}
    with mp.Pool(
        processes=max(1, args.num_workers),
        initializer=_init_worker,
        initargs=(folder_path, args.model_path, args.conv_mode, args.query, args.precision)
    ) as pool, tqdm(total=len(files), desc=f"Processing {os.path.basename(folder_path)}", unit="video") as pbar:
        for batch_len, batch_dict in pool.imap_unordered(_process_batch, batches, chunksize=1):
            # cập nhật thanh tiến trình theo SỐ VIDEO trong batch
            pbar.update(batch_len)
            results.update(batch_dict)

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{os.path.basename(os.path.normpath(folder_path))}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved {len(results)} entries to: {out_path}")

if __name__ == "__main__":
    main_entry()
