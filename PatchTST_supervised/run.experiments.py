import os
import subprocess
import re
import csv
import itertools
import sys
import torch 
import math
from collections import defaultdict
from datetime import datetime

# ==================================================================================
# [1] 데이터셋별 공식 하이퍼파라미터
# ==================================================================================
HYPER_PARAMS = {
    'etth1': {
        'enc_in': 7, 'e_layers': 3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128,
        'dropout': 0.3, 'fc_dropout': 0.3, 'head_dropout': 0,
        'train_epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001,
        'patience': 100
    },
    'etth2': {
        'enc_in': 7, 'e_layers': 3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128,
        'dropout': 0.3, 'fc_dropout': 0.3, 'head_dropout': 0,
        'train_epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001,
        'patience': 100
    },
    'ettm1': {
        'enc_in': 7, 'e_layers': 3, 'n_heads': 16, 'd_model': 128, 'd_ff': 256,
        'dropout': 0.2, 'fc_dropout': 0.2, 'head_dropout': 0,
        'train_epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001,
        'patience': 20, 'lradj': 'TST', 'pct_start': 0.4
    },
    'ettm2': {
        'enc_in': 7, 'e_layers': 3, 'n_heads': 16, 'd_model': 128, 'd_ff': 256,
        'dropout': 0.2, 'fc_dropout': 0.2, 'head_dropout': 0,
        'train_epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001,
        'patience': 20, 'lradj': 'TST', 'pct_start': 0.4
    },
    'electricity': {
        'enc_in': 321, 'e_layers': 3, 'n_heads': 16, 'd_model': 128, 'd_ff': 256,
        'dropout': 0.2, 'fc_dropout': 0.2, 'head_dropout': 0,
        'train_epochs': 100, 'batch_size': 32, 'learning_rate': 0.0001,
        'patience': 10, 'lradj': 'TST', 'pct_start': 0.2
    },
    'traffic': {
        'enc_in': 862, 'e_layers': 3, 'n_heads': 16, 'd_model': 128, 'd_ff': 256,
        'dropout': 0.2, 'fc_dropout': 0.2, 'head_dropout': 0,
        'train_epochs': 100, 'batch_size': 24, 'learning_rate': 0.0001,
        'patience': 10, 'lradj': 'TST', 'pct_start': 0.2
    },
    'weather': {
        'enc_in': 21, 'e_layers': 3, 'n_heads': 16, 'd_model': 128, 'd_ff': 256,
        'dropout': 0.2, 'fc_dropout': 0.2, 'head_dropout': 0,
        'train_epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001,
        'patience': 20
    }
}

# ==================================================================================
# [2] 실험 변수 설정
# ==================================================================================

DATASETS = {
    'etth1': 'ETTh1.csv',
    # 'etth2': 'ETTh2.csv',
      'ettm1': 'ETTm1.csv',
    # 'ettm2': 'ETTm2.csv',
    # 'electricity': 'electricity.csv',    
    #  'traffic': 'traffic.csv',
      'weather': 'weather.csv',
}

# PATCH_LENS는 이제 "Base Patch Length" (가장 작은 패치 길이)를 의미합니다.
PATCH_LENS = [16] 
PRED_LENS = [96, 192, 336, 720]

# Scales 설정: [1]은 기존 PatchTST와 동일, [1, 2]는 16, 32 길이 동시 사용
SCALES_LIST = [
    [1], 
    [1, 2], 
    [1, 4],
    [1, 3, 5], 
    [1, 2, 3],
    [1, 4, 8]
]

SEQ_LEN = 336
MODEL_NAME = 'PatchTST' # Multi-Scale 코드가 PatchTST 클래스로 덮어씌워졌다면 그대로 사용
RESULT_FILE = 'experiment_results_multi_halfstride_patchlength_info.csv'
LOG_DIR = './logs_multi_halfstride_patchlength_info/'
RESULT_COLUMNS = [
    'Timestamp', 'Dataset', 'Seq_Len', 'Pred_Len',
    'Base_Patch_Len', 'Strides', 'Patch_Counts', 'Scales',
    'MSE', 'MAE', 'Len_Alpha', 'Cross_Alpha', 'Log_File', 'Hyperparams'
]

# ==================================================================================
# [3] 유틸리티 함수
# ==================================================================================

def compute_scale_patch_num(seq_len, base_patch_len, patch_len, stride, padding_patch='end'):
    # Match MultiScalePatchTST_backbone._compute_patch_num
    pad_left = (patch_len - base_patch_len) // 2
    pad_right = stride if padding_patch == 'end' else 0
    padded_len = seq_len + pad_left + pad_right
    if padded_len < patch_len:
        return 1
    return int((padded_len - patch_len) / stride + 1)

def build_scale_info(seq_len, base_patch_len, scales, padding_patch='end'):
    patch_lens = [base_patch_len * s for s in scales]
    strides = [max(1, pl // 2) for pl in patch_lens]
    patch_counts = [
        compute_scale_patch_num(seq_len, base_patch_len, pl, st, padding_patch=padding_patch)
        for pl, st in zip(patch_lens, strides)
    ]
    return patch_lens, strides, patch_counts

def parse_metrics(output_str):
    try:
        match = re.search(r"mse:([\d\.]+), mae:([\d\.]+)", output_str)
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception:
        pass
    return None, None

def parse_alphas(output_str):
    # Expected line from run_longExp.py:
    # ALPHAS len_alpha=<value> cross_alpha=<value>
    try:
        match = re.search(r"ALPHAS\s+len_alpha=([-\deE\.+]+)\s+cross_alpha=([-\deE\.+]+)", output_str)
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception:
        pass
    return None, None

def get_data_args(dataset_name):
    d_lower = dataset_name.lower()
    
    # 1. ETT Series
    if d_lower.startswith('ett'):
        if d_lower == 'etth1': d_code = 'ETTh1'
        elif d_lower == 'etth2': d_code = 'ETTh2'
        elif d_lower == 'ettm1': d_code = 'ETTm1'
        elif d_lower == 'ettm2': d_code = 'ETTm2'
        else: d_code = 'ETTh1'
        
        return {
            'data': d_code,
            'root_path': './dataset/ETT/', 
            'features': 'M'
        }
    
    # 2. Custom Series
    else:
        return {
            'data': 'custom',
            'root_path': f'./dataset/{d_lower}/', 
            'features': 'M'
        }

def to_float_or_none(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x

def _pick_field(row, *keys):
    for key in keys:
        val = row.get(key)
        if val not in (None, ''):
            return val
    return ''

def ensure_result_file_schema(path):
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(RESULT_COLUMNS)
        return

    with open(path, mode='r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(RESULT_COLUMNS)
        return

    header = rows[0]
    if header == RESULT_COLUMNS:
        return

    with open(path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)

    upgraded_rows = []
    for row in old_rows:
        upgraded_rows.append([
            _pick_field(row, 'Timestamp', 'timestamp'),
            _pick_field(row, 'Dataset', 'dataset'),
            _pick_field(row, 'Seq_Len', 'seq_len'),
            _pick_field(row, 'Pred_Len', 'pred_len'),
            _pick_field(row, 'Base_Patch_Len', 'base_patch_len'),
            _pick_field(row, 'Strides', 'Stride', 'stride', 'strides'),
            _pick_field(
                row,
                'Patch_Counts',
                'Patch_Count',
                'patch_count_scales',
                'patch_count',
                'Patch_Count_Scales'
            ),
            _pick_field(row, 'Scales', 'scales'),
            _pick_field(row, 'MSE', 'mse'),
            _pick_field(row, 'MAE', 'mae'),
            _pick_field(row, 'Len_Alpha', 'len_alpha'),
            _pick_field(row, 'Cross_Alpha', 'cross_alpha'),
            _pick_field(row, 'Log_File', 'log_file'),
            _pick_field(row, 'Hyperparams', 'hyperparameter', 'hyperparams')
        ])

    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_COLUMNS)
        writer.writerows(upgraded_rows)

    print(f"Upgraded result CSV schema: {path}")

def append_scale_avg_rows(run_records):
    # Group key: (Dataset, Base_Patch_Len, Scales)
    grouped = defaultdict(list)
    for rec in run_records:
        key = (rec['dataset'], rec['base_patch_len'], rec['scales'])
        grouped[key].append(rec)

    pred_len_cols = sorted(PRED_LENS)
    with open(RESULT_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)

        for (dataset, base_patch_len, scales), rows in sorted(grouped.items()):
            mse_vals = []
            mae_vals = []
            len_alpha_vals = []
            cross_alpha_vals = []
            row_by_pl = {r['pred_len']: r for r in rows}
            completed = 0

            # collect average over available pred_lens
            details = []
            for pl in pred_len_cols:
                r = row_by_pl.get(pl)
                if r is None:
                    continue
                mse = to_float_or_none(r['mse'])
                mae = to_float_or_none(r['mae'])
                if mse is not None:
                    mse_vals.append(mse)
                if mae is not None:
                    mae_vals.append(mae)
                if (mse is not None) or (mae is not None):
                    completed += 1
                len_a = to_float_or_none(r['len_alpha'])
                cross_a = to_float_or_none(r['cross_alpha'])
                if len_a is not None:
                    len_alpha_vals.append(len_a)
                if cross_a is not None:
                    cross_alpha_vals.append(cross_a)
                details.append(
                    f"pl{pl}:mse={r['mse']},mae={r['mae']},len_alpha={r['len_alpha']},cross_alpha={r['cross_alpha']}"
                )

            avg_mse = sum(mse_vals) / len(mse_vals) if mse_vals else float('nan')
            avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else float('nan')
            avg_len_alpha = sum(len_alpha_vals) / len(len_alpha_vals) if len_alpha_vals else float('nan')
            avg_cross_alpha = sum(cross_alpha_vals) / len(cross_alpha_vals) if cross_alpha_vals else float('nan')
            strides_safe = rows[0]['strides']
            patch_counts_safe = rows[0]['patch_counts']
            detail_str = f"AVG_BY_PRED_LEN;completed={completed};" + ";".join(details)

            # Reuse the existing RESULT_FILE schema and append one avg row per scale combo.
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                dataset, SEQ_LEN, "AVG",
                base_patch_len, strides_safe, patch_counts_safe, scales,
                avg_mse, avg_mae, avg_len_alpha, avg_cross_alpha, "SUMMARY", detail_str
            ])

# ==================================================================================
# [4] 메인 실행 루프
# ==================================================================================

def run_experiments():
    print("=" * 60)
    print("Checking GPU Environment...")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"❌ GPU is NOT available. Running on CPU.")
    print("=" * 60)
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    ensure_result_file_schema(RESULT_FILE)

    combinations = list(itertools.product(DATASETS.keys(), PATCH_LENS, PRED_LENS, SCALES_LIST))
    total_exps = len(combinations)
    run_records = []
    
    print(f"\nTotal Experiments: {total_exps}")
    
    for idx, (dataset_name, p_len, pred_len, scales) in enumerate(combinations):
        d_file = DATASETS[dataset_name]
        data_config = get_data_args(dataset_name)
        patch_lens, strides, patch_counts = build_scale_info(SEQ_LEN, p_len, scales, padding_patch='end')
        stride = strides[0]
        
        scales_str = '_'.join(map(str, scales))
        model_id = f"{dataset_name}_pl{pred_len}_base{p_len}_sc{scales_str}"
        log_filename = f"{model_id}.log"
        log_path = os.path.join(LOG_DIR, log_filename)
        
        hyper_params = HYPER_PARAMS.get(dataset_name, {})
        
        # Command 구성
        cmd = [
            "python", "-u", "run_longExp.py",
            "--is_training", "1",
            "--model_id", model_id,
            "--model", MODEL_NAME,
            "--data", data_config['data'],
            "--root_path", data_config['root_path'],
            "--data_path", d_file,
            "--features", data_config['features'],
            "--seq_len", str(SEQ_LEN),
            "--pred_len", str(pred_len),
            "--patch_len", str(p_len),  # 이것이 Base Patch Length가 됨
            "--stride", str(stride),
            "--des", "Exp",
            "--itr", "1",
            "--use_amp"
        ]

        # Add Scales Argument (list로 전달)
        cmd.append("--scales")
        cmd.extend(map(str, scales))

        for key, value in hyper_params.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))
        
        print(f"\n[{idx+1}/{total_exps}] Running: {model_id}")
        # print(f"Command: {' '.join(cmd)}") # 디버깅용 커맨드 출력

        full_output = ""
        try:
            with open(log_path, "w", encoding="utf-8") as log_f:
                if torch.cuda.is_available():
                    log_f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write("-" * 50 + "\n")

                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    bufsize=1
                )
                for line in process.stdout:
                    log_f.write(line)
                    full_output += line
                    # 진행상황을 터미널에도 간략히 출력하고 싶다면 주석 해제
                    # print(line, end='') 
                process.wait()

            mse, mae = parse_metrics(full_output)
            len_alpha, cross_alpha = parse_alphas(full_output)
            
            if mse is not None:
                print(f"   > Success! MSE={mse}, MAE={mae}")
            else:
                print(f"   > Warning: Metrics not found. Check log.")
                mse, mae = "NaN", "NaN"
            if len_alpha is None or cross_alpha is None:
                print("   > Warning: Alpha values not found. Check log.")
                len_alpha, cross_alpha = "NaN", "NaN"

            scales_safe = str(scales).replace(',', ';')
            strides_safe = str(strides).replace(',', ';')
            patch_counts_safe = str(patch_counts).replace(',', ';')
            params_safe = str(hyper_params).replace(',', ';')

            with open(RESULT_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    dataset_name, SEQ_LEN, pred_len,
                    p_len, strides_safe, patch_counts_safe, scales_safe,
                    mse, mae, len_alpha, cross_alpha, log_filename, params_safe
                ])

            run_records.append({
                'dataset': dataset_name,
                'base_patch_len': p_len,
                'pred_len': pred_len,
                'scales': scales_safe,
                'strides': strides_safe,
                'patch_counts': patch_counts_safe,
                'mse': mse,
                'mae': mae,
                'len_alpha': len_alpha,
                'cross_alpha': cross_alpha,
            })
                
        except Exception as e:
            print(f"   > Python Script Error: {e}")
            continue

    if run_records:
        append_scale_avg_rows(run_records)
        print(f"\nAverage rows appended to: {RESULT_FILE}")

    print(f"\nAll experiments finished!")

if __name__ == "__main__":
    run_experiments()
