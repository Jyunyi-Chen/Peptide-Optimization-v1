import os
import torch as T
import numpy as np
import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tf_keras as keras

_AVP_DIR = "./avp_prediction"
_MODEL_N = os.path.join(_AVP_DIR, "ai4avp_n_model.h5")
_PC6_PATH = os.path.join(_AVP_DIR, "6-pc")
_PAD_LEN = 50

def _build_pc6_table() -> dict[str, list[float]]:

    df = pd.read_csv(_PC6_PATH, sep=" ", index_col=0)

    def _zscore(col: pd.Series) -> np.ndarray:
        return (col - col.mean()) / col.std(ddof=1)

    matrix = np.array([
        _zscore(df["H1"]),
        _zscore(df["V"]),
        _zscore(df["P1"]),
        _zscore(df["Pl"]),
        _zscore(df["PKa"]),
        _zscore(df["NCI"]),
    ])

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    table: dict[str, list[float]] = {
        aa: matrix[:, i].tolist()
        for i, aa in enumerate(amino_acids)
    }
    table["X"] = [0.0] * 6

    return table

ENCODING_TABLE: dict[str, list[float]] = _build_pc6_table()
MODEL_N = keras.models.load_model(_MODEL_N, compile=False)
DEVICE = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

def batch_encode_peps(peptides: list[str], length: int = _PAD_LEN) -> np.ndarray:

    pep_vectors: list[list[list[float]]] = []

    for pep in peptides:
        pep = pep.ljust(length, "X")[:length]
        pep_vectors.append([ENCODING_TABLE.get(aa, [0.0] * 6) for aa in pep])

    return np.array(pep_vectors, dtype=np.float32)

def get_avp_probs(peptides: list[str]) -> T.Tensor:

    scores: np.ndarray = MODEL_N.predict(batch_encode_peps(peptides), verbose=0, batch_size=512)

    return T.tensor(scores, dtype=T.float32, device=DEVICE).squeeze()

if __name__ == "__main__":

    import time

    for _ in range(3): get_avp_probs(["FLTHILKGLFTAGKVIHGLIHRRRHG"] * 10)

    peptides = ["FLTHILKGLFTAGKVIHGLIHRRRHG"] * 500

    times = []
    for _ in range(20):

        t1 = time.time()
        get_avp_probs(peptides)
        t2 = time.time()

        times.append(t2 - t1)

    print(f"Stable inference time for 500 sequences: {np.mean(times):.4f} ± {np.std(times):.4f} sec")

    print(f"{get_avp_probs(['SIGTAVKKAVPIAKKVGKVAIPIAKAVLSVVGQLVG']).item():.8f}")
    print(f"{get_avp_probs(['FLSLLPSIVSGAVSLAKKLG']).item():.8f}")
    print(f"{get_avp_probs(['FLPIIAKLLGGLL']).item():.8f}")
    print(f"{get_avp_probs(['FLPIPRPILLGLL']).item():.8f}")
    print(f"{get_avp_probs(['FLIIRRPIVLGLL']).item():.8f}")
    
    print(f"{get_avp_probs(['GLHKVMREVLGYERNSYKKFFLR']).item():.8f}")
    print(f"{get_avp_probs(['INWKKIAEVGGKILSSL']).item():.8f}")
    print(f"{get_avp_probs(['INWKGIAAMAKKLL']).item():.8f}")
    print(f"{get_avp_probs(['INWKKIAEIGKQVLSAL']).item():.8f}")
    print(f"{get_avp_probs(['FLPMLAGLAANFLPKLFCKITKKC']).item():.8f}")