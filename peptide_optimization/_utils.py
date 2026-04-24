import os
import json
import types
import config
import random
import torch as T
import numpy as np
import importlib.util


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from tokenizers import Tokenizer
from scipy.ndimage import gaussian_filter1d
from huggingface_hub import hf_hub_download

PEPBERT_ID2MODEL = \
{
    "dzjxzyd/PepBERT-small-UniParc"  : "tmodel_16.pt", "dzjxzyd/PepBERT-large-UniParc"  : "tmodel_17.pt", 
    "dzjxzyd/PepBERT-small-UniRef50" : "tmodel_14.pt", "dzjxzyd/PepBERT-large-UniRef50" : "tmodel_16.pt", 
    "dzjxzyd/PepBERT-small-UniRef90" : "tmodel_18.pt", "dzjxzyd/PepBERT-large-UniRef90" : "tmodel_13.pt", 
    "dzjxzyd/PepBERT-small-UniRef100": "tmodel_12.pt", "dzjxzyd/PepBERT-large-UniRef100": "tmodel_17.pt",
}

def set_seeds() -> None:

    random.seed(config.RANDOM_SEED)
    T.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    if T.cuda.is_available():
        T.cuda.manual_seed(config.RANDOM_SEED)
        T.cuda.manual_seed_all(config.RANDOM_SEED)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False

def get_save_dir() -> str:

    today = datetime.today().strftime("%Y-%m-%d")
    logs_dir = os.path.join("./peptide_optimization", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    prefix = f"{today}-"
    existing = [d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d)) and d.startswith(prefix)]

    nums = [int(d.split("-")[-1]) for d in existing 
            if d.split("-")[-1].isdigit()]
    next_num = max(nums, default=0) + 1

    dir_name = f"{today}-{next_num:02d}"
    save_dir = os.path.join(logs_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_config(module: types.ModuleType, save_dir: str):

    config_dict = {k: v for k, v in vars(module).items() 
                   if not k.startswith("__") and not callable(v)}
    
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

def get_pepbert(repo_id: str = "dzjxzyd/PepBERT-large-UniParc"):

    model_module = load_module(repo_id, "model.py")
    build_transformer = model_module.build_transformer
    config_module = load_module(repo_id, "config.py")
    get_config = config_module.get_config

    tokenizer_path = hf_hub_download(repo_id, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    config = get_config()
    model: T.nn.Module = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=config["seq_len"],
        d_model=config["d_model"])

    weights_path = hf_hub_download(repo_id, PEPBERT_ID2MODEL[repo_id])
    device = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")
    state = T.load(weights_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    return model, tokenizer

def load_module(repo_id: str, file_name: str) -> any:

    file_path = hf_hub_download(repo_id, file_name)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def plot_single_smooth(y_data: list[float], x_label: str, y_label: str, title: str, 
        fig_name: str, save_dir: str, sigma: int = 200) -> None:

    plt.figure(figsize=(10, 6))
    x_data = range(1, len(y_data) + 1)
    plt.plot(x_data, gaussian_filter1d(y_data, sigma=sigma))

    plt.xlabel(x_label, labelpad=10)
    plt.ylabel(y_label, labelpad=10)
    plt.title(title, pad=10)
    plt.grid(True)

    fig_path = os.path.join(save_dir, f"{fig_name}.svg")
    plt.savefig(fig_path)
    plt.close()

def plot_multip_smooth(y_data: list[list[float]], x_label: str, y_label: str, title: str, 
        fig_name: str, legends: list[str], save_dir: str, sigma: int = 200) -> None:
    
    plt.figure(figsize=(10, 6))
    x_data = range(1, len(y_data[0]) + 1)
    for idx, series in enumerate(y_data):
        plt.plot(x_data, gaussian_filter1d(series, sigma=sigma), label=legends[idx])

    plt.xlabel(x_label, labelpad=10)
    plt.ylabel(y_label, labelpad=10)
    plt.title(title, pad=10)
    plt.grid(True)
    plt.legend()

    fig_path = os.path.join(save_dir, f"{fig_name}.svg")
    plt.savefig(fig_path)
    plt.close()