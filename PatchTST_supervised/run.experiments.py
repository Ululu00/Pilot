import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
from datetime import datetime

# ==================================================================================
# [1] Dataset Hyperparameters
# ==================================================================================
HYPER_PARAMS = {
    'etth1': {
        'enc_in': 7, 'e_layers': 1, 'n_heads': 8, 'd_model': 256, 'd_ff': 2048,
        'dropout': 0.1,
        'train_epochs': 10, 'batch_size': 4, 'learning_rate': 0.0001,
        'patience': 3, 'lradj': 'type1'
    },
    'etth2': {
        'enc_in': 7, 'e_layers': 1, 'n_heads': 8, 'd_model': 256, 'd_ff': 1024,
        'dropout': 0.1,
        'train_epochs': 10, 'batch_size': 16, 'learning_rate': 0.0001,
        'patience': 3, 'lradj': 'type1'
    },
    'ettm1': {
        'enc_in': 7, 'e_layers': 1, 'n_heads': 8, 'd_model': 256, 'd_ff': 2048,
        'dropout': 0.1,
        'train_epochs': 10, 'batch_size': 4, 'learning_rate': 0.0001,
        'patience': 3, 'lradj': 'type1'
    },
    'ettm2': {
        'enc_in': 7, 'e_layers': 1, 'n_heads': 8, 'd_model': 256, 'd_ff': 2048,
        'dropout': 0.1,
        'train_epochs': 10, 'batch_size': 32, 'learning_rate': 0.0001,
        'patience': 3, 'lradj': 'type1'
    },
    'electricity': {
        'enc_in': 321, 'e_layers': 4, 'n_heads': 8, 'd_model': 512, 'd_ff': 512,
        'dropout': 0.1,
        'train_epochs': 10, 'batch_size': 4, 'learning_rate': 0.0001,
        'patience': 3, 'lradj': 'type1'
    },
    'traffic': {
        'enc_in': 862, 'e_layers': 3, 'n_heads': 8, 'd_model': 512, 'd_ff': 512,
        'dropout': 0.1,
        'train_epochs': 10, 'batch_size': 16, 'learning_rate': 0.001,
        'patience': 3, 'lradj': 'type1'
    },
    'weather': {
        'enc_in': 21, 'e_layers': 1, 'n_heads': 8, 'd_model': 256, 'd_ff': 512,
        'dropout': 0.1,
        'train_epochs': 10, 'batch_size': 4, 'learning_rate': 0.0001,
        'patience': 3, 'lradj': 'type1'
    }
}


# ==================================================================================
# [2] Baseline Experiment Config
# ==================================================================================
DATASETS = {
    'etth1': 'ETTh1.csv',
    'ettm1': 'ETTm1.csv',
    'weather': 'weather.csv',
}

PATCH_LENS = [16]
PRED_LENS = [96, 192, 336, 720]

SEQ_LEN = 96
SCALES = [1, 3, 5]
STRIDE_STRATEGY = 'half'
PE_MODE = 'rope_abs'

MODEL_NAME = 'PatchTST'
RESULT_FILE = 'experiment_results_baseline.csv'
LOG_DIR = './logs_baseline/'
LOCK_FILE = '.run_experiments_baseline.lock'

RESULT_COLUMNS = [
    'Timestamp', 'Dataset', 'Seq_Len', 'Pred_Len',
    'Base_Patch_Len', 'Strides', 'Patch_Counts', 'Scales',
    'MSE', 'MAE', 'Log_File', 'Hyperparams'
]


# ==================================================================================
# [3] Utility Helpers
# ==================================================================================
def compute_scale_patch_num(seq_len, base_patch_len, patch_len, stride, padding_patch='end'):
    pad_left = (patch_len - base_patch_len) // 2
    pad_right = stride if padding_patch == 'end' else 0
    padded_len = seq_len + pad_left + pad_right
    if padded_len < patch_len:
        return 1
    return int((padded_len - patch_len) / stride + 1)


def build_scale_info(seq_len, base_patch_len, scales, padding_patch='end'):
    patch_lens = [base_patch_len * s for s in scales]
    strides = [max(1, int(round(pl * 0.5))) for pl in patch_lens]
    patch_counts = [
        compute_scale_patch_num(seq_len, base_patch_len, patch_len, stride, padding_patch=padding_patch)
        for patch_len, stride in zip(patch_lens, strides)
    ]
    return patch_lens, strides, patch_counts


def compose_model_id(dataset_name, pred_len, base_patch_len):
    scales_str = '_'.join(map(str, SCALES))
    return f"{dataset_name}_sl{SEQ_LEN}_pl{pred_len}_base{base_patch_len}_sc{scales_str}"


def parse_metrics(output_str):
    try:
        match = re.search(r"mse:([\d\.]+), mae:([\d\.]+)", output_str)
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception:
        pass
    return None, None


def is_worker_segfault(output_str):
    lower = output_str.lower()
    return (
        "unexpected segmentation fault encountered in worker" in lower
        or ("dataloader worker" in lower and "segmentation fault" in lower)
    )


def upsert_cmd_arg(cmd, key, value):
    updated = list(cmd)
    if key in updated:
        idx = updated.index(key)
        if idx + 1 < len(updated):
            updated[idx + 1] = str(value)
        else:
            updated.append(str(value))
        return updated
    updated.extend([key, str(value)])
    return updated


def run_and_capture(cmd, log_path, append=False, prefix_note=None):
    full_output = ""
    mode = 'a' if append else 'w'
    with open(log_path, mode, encoding='utf-8', buffering=1) as log_f:
        if not append:
            import torch
            if torch.cuda.is_available():
                log_f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        else:
            if prefix_note:
                log_f.write("\n" + "=" * 20 + f" {prefix_note} " + "=" * 20 + "\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write("-" * 50 + "\n")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log_f.write(line)
            log_f.flush()
            full_output += line
        process.wait()

    return full_output, process.returncode


def build_combinations():
    return list(itertools.product(DATASETS.keys(), PATCH_LENS, PRED_LENS))


def get_data_args(dataset_name):
    d_lower = dataset_name.lower()

    if d_lower.startswith('ett'):
        if d_lower == 'etth1':
            d_code = 'ETTh1'
        elif d_lower == 'etth2':
            d_code = 'ETTh2'
        elif d_lower == 'ettm1':
            d_code = 'ETTm1'
        elif d_lower == 'ettm2':
            d_code = 'ETTm2'
        else:
            d_code = 'ETTh1'
        return {
            'data': d_code,
            'root_path': './dataset/ETT/',
            'features': 'M',
        }

    return {
        'data': 'custom',
        'root_path': f'./dataset/{d_lower}/',
        'features': 'M',
    }


def ensure_result_file_schema(path):
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as f:
            csv.writer(f).writerow(RESULT_COLUMNS)
        return

    with open(path, mode='r', newline='') as f:
        rows = list(csv.reader(f))

    if not rows:
        with open(path, mode='w', newline='') as f:
            csv.writer(f).writerow(RESULT_COLUMNS)
        return

    header = rows[0]
    if header != RESULT_COLUMNS:
        raise ValueError(
            f"Existing result file has unexpected header: {path}\n"
            f"Expected: {RESULT_COLUMNS}\n"
            f"Found: {header}"
        )


def _pid_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                pid = int(f.read().strip())
        except Exception:
            pid = -1
        if _pid_alive(pid):
            raise RuntimeError(f"another run.experiments.py is already running (pid={pid})")
        try:
            os.remove(path)
        except OSError:
            pass
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(os.getpid()))


def release_run_lock(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ==================================================================================
# [4] Main Loop
# ==================================================================================
def run_experiments(start_model_id=None, num_workers_override=None):
    import torch

    acquire_run_lock(LOCK_FILE)
    try:
        print("=" * 60)
        print("Checking GPU Environment...")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU is NOT available. Running on CPU.")
        print("=" * 60)

        os.makedirs(LOG_DIR, exist_ok=True)
        ensure_result_file_schema(RESULT_FILE)

        combinations = build_combinations()
        total_exps = len(combinations)
        print(f"\nTotal Experiments: {total_exps}")

        start_idx = 0
        if start_model_id:
            found_idx = -1
            for idx, (dataset_name, patch_len, pred_len) in enumerate(combinations):
                model_id = compose_model_id(dataset_name, pred_len, patch_len)
                if model_id == start_model_id:
                    found_idx = idx
                    break
            if found_idx < 0:
                raise ValueError(f"start_model_id not found in combinations: {start_model_id}")
            start_idx = found_idx
            print(f"Resuming from [{start_idx + 1}/{total_exps}] {start_model_id}")

        for idx in range(start_idx, total_exps):
            dataset_name, patch_len, pred_len = combinations[idx]
            data_file = DATASETS[dataset_name]
            data_config = get_data_args(dataset_name)
            _, strides, patch_counts = build_scale_info(SEQ_LEN, patch_len, SCALES, padding_patch='end')
            stride = strides[0]

            model_id = compose_model_id(dataset_name, pred_len, patch_len)
            log_filename = f"{model_id}.log"
            log_path = os.path.join(LOG_DIR, log_filename)
            hyper_params = HYPER_PARAMS.get(dataset_name, {})

            cmd = [
                sys.executable, "-u", "run_longExp.py",
                "--is_training", "1",
                "--model_id", model_id,
                "--model", MODEL_NAME,
                "--data", data_config['data'],
                "--root_path", data_config['root_path'],
                "--data_path", data_file,
                "--features", data_config['features'],
                "--seq_len", str(SEQ_LEN),
                "--pred_len", str(pred_len),
                "--patch_len", str(patch_len),
                "--stride", str(stride),
                "--stride_strategy", STRIDE_STRATEGY,
                "--pe", PE_MODE,
                "--des", "Exp",
                "--itr", "1",
                "--use_amp",
                "--scales",
                *map(str, SCALES),
            ]
            if num_workers_override is not None:
                cmd.extend(["--num_workers", str(num_workers_override)])

            for key, value in hyper_params.items():
                cmd.extend([f"--{key}", str(value)])

            print(f"\n[{idx + 1}/{total_exps}] Running: {model_id}")

            try:
                full_output, retcode = run_and_capture(cmd, log_path, append=False)

                if retcode != 0 and is_worker_segfault(full_output):
                    print("   > Worker segmentation fault detected. Retrying with --num_workers 0 ...")
                    retry_cmd = upsert_cmd_arg(cmd, "--num_workers", "0")
                    retry_output, retry_retcode = run_and_capture(
                        retry_cmd, log_path, append=True, prefix_note="RETRY_WITH_NUM_WORKERS_0"
                    )
                    full_output += "\n" + retry_output
                    retcode = retry_retcode

                mse, mae = parse_metrics(full_output)
                if retcode == 0 and mse is not None:
                    print(f"   > Success! MSE={mse}, MAE={mae}")
                else:
                    if retcode != 0:
                        print(f"   > Warning: process exited with code {retcode}. Check log.")
                    else:
                        print("   > Warning: Metrics not found. Check log.")
                    mse, mae = "NaN", "NaN"

                with open(RESULT_FILE, mode='a', newline='') as f:
                    csv.writer(f).writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        dataset_name,
                        SEQ_LEN,
                        pred_len,
                        patch_len,
                        str(strides).replace(',', ';'),
                        str(patch_counts).replace(',', ';'),
                        str(SCALES).replace(',', ';'),
                        mse,
                        mae,
                        log_filename,
                        str(hyper_params).replace(',', ';'),
                    ])
            except Exception as e:
                print(f"   > Python Script Error: {e}")
                continue

        print("\nAll experiments finished!")
        return 'finished'
    finally:
        release_run_lock(LOCK_FILE)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the fixed baseline PatchTST experiment sweep.')
    parser.add_argument(
        '--start_model_id',
        type=str,
        default='',
        help='resume from this model_id (inclusive)',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='override DataLoader num_workers for run_longExp.py',
    )

    raw_argv = sys.argv[1:]
    cleaned_argv = []
    for tok in raw_argv:
        tok_clean = tok.replace('\u00A0', ' ').strip()
        if tok_clean:
            cleaned_argv.append(tok_clean)

    args, unknown = parser.parse_known_args(cleaned_argv)
    unknown = [u for u in unknown if u.replace('\u00A0', ' ').strip()]
    if unknown:
        parser.error("unrecognized arguments: " + " ".join(repr(u) for u in unknown))
    return args


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        start_model_id=(args.start_model_id.strip() or None),
        num_workers_override=args.num_workers,
    )
