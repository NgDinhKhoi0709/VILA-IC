import os, json, argparse, multiprocessing as mp
from functools import lru_cache
from natsort import natsorted
from tqdm import tqdm
import transformers, torch
from huggingface_hub import snapshot_download

transformers.logging.set_verbosity_error()

from llava.eval.run_vila import main, load_model_once

# ===== Globals cho worker =====
G_FOLDER_PATH = G_MODEL_PATH = G_CONV_MODE = G_QUERY = None
G_TOKENIZER = G_MODEL = G_IMAGE_PROCESSOR = None
PRECISION = "fp16"  # default

@lru_cache(maxsize=None)
def _load_model_cached(model_path: str, conv_mode: str):
    tokenizer, model, image_processor = load_model_once(model_path, conv_mode)
    # Apply precision
    if PRECISION == "fp16":
        model = model.half()
    elif PRECISION == "bf16":
        model = model.to(torch.bfloat16)
    # Enable TF32 for Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tokenizer, model, image_processor

def _init_worker(folder_path, model_path, conv_mode, query, precision):
    global G_FOLDER_PATH, G_MODEL_PATH, G_CONV_MODE, G_QUERY
    global G_TOKENIZER, G_MODEL, G_IMAGE_PROCESSOR, PRECISION
    G_FOLDER_PATH, G_MODEL_PATH, G_CONV_MODE, G_QUERY = folder_path, model_path, conv_mode, query
    PRECISION = precision
    # Offline mode to prevent HF requests
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    G_TOKENIZER, G_MODEL, G_IMAGE_PROCESSOR = _load_model_cached(G_MODEL_PATH, G_CONV_MODE)

def extract_id_from_filename(file_name: str) -> str:
    try:
        return file_name.split('_')[1].split('.')[0]
    except Exception:
        return os.path.splitext(file_name)[0]

def _process_one(fname: str):
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
        vid = extract_id_from_filename(fname)
        return (vid, out.strip() if out else None)
    except Exception:
        return (extract_id_from_filename(fname), None)

def main_entry():
    try:
        mp.set_start_method("spawn", force=True)  # fix CUDA + fork
    except RuntimeError:
        pass

    p = argparse.ArgumentParser(description="Process videos (multiprocessing, fp16/bf16/fp32, preload offline).")
    p.add_argument('--folder_path', required=True)
    p.add_argument('--output_path', default='/kaggle/working/output')
    p.add_argument('--model_path', default='Efficient-Large-Model/VILA1.5-3b')
    p.add_argument('--conv_mode', default='vicuna_v1')
    p.add_argument('--query', default='<video>\n Please describe the video in detail!')
    p.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2)//2))
    p.add_argument('--precision', choices=['fp16','bf16','fp32'], default='fp16')
    p.add_argument('--hf_token', default=None, help='Hugging Face token (optional)')
    args = p.parse_args()

    folder_path = args.folder_path
    output_folder = args.output_path

    # ===== Preload model to local cache =====
    print(f"📥 Downloading model {args.model_path} to local cache...")
    local_model_path = snapshot_download(
        repo_id=args.model_path,
        token=args.hf_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        local_dir_use_symlinks=False
    )
    print(f"✅ Model cached at: {local_model_path}")

    # ===== Enable offline mode globally =====
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    exts = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
    files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])
    if not files:
        print(f"No video files found in folder: {folder_path}")
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=max(1, args.num_workers),
        initializer=_init_worker,
        initargs=(folder_path, local_model_path, args.conv_mode, args.query, args.precision)
    ) as pool:
        it = pool.imap_unordered(_process_one, files, chunksize=1)
        results = {}
        for vid, text in tqdm(it, total=len(files), desc=f"Processing {os.path.basename(folder_path)}", unit="video"):
            if text is not None:
                results[vid] = text

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{os.path.basename(os.path.normpath(folder_path))}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved {len(results)} entries to: {out_path}")

if __name__ == "__main__":
    main_entry()
