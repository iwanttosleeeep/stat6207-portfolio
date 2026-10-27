"""Transformer experiment - STAGE 1: validation-only tuning (train rows 0-639, val rows 640-799).

STRICT: this script must never touch test rows 800-999. Asserts enforce this.
All decisions (features, lookback, architecture, hyperparams, batch order) use
training + validation only.
"""
import math, os, random, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.use_deterministic_algorithms(True)
set_seed()

DATA = "A2_data_release.csv"
N_TRAIN_END, N_VAL_END = 640, 800  # positional splits; test 800-1000 untouched

# ---------- load & verify ----------
df = pd.read_csv(DATA)
assert len(df) == 1000, len(df)
assert df.isna().sum().sum() == 0
assert (df["rv1"] == df["rv2"]).all(), "rv1/rv2 assumed identical"
assert (pd.to_datetime(df["date"]).diff().dropna() == pd.Timedelta("10min")).all()

TARGET = "Appliances"
ALL_COLS = [c for c in df.columns if c not in ("date", "rv2", TARGET)]  # exogenous, rv2 dropped (dup of rv1)

# Feature-subset candidates (compared on validation only)
FEAT_SETS = {
    # univariate baseline: tests how much exogenous history helps
    "F1_univariate": [TARGET],
    # compact physics-informed: target + lights + kitchen T/RH + outdoor drivers + solar proxy.
    # Drops redundant room sensors (T2-T9/RH_2-RH_9), pressure, visibility to limit params (n=1000).
    "F2_compact": [TARGET, "lights", "T1", "RH_1", "T_out", "RH_out", "Windspeed", "Tdewpoint", "rv1"],
    # full (dedup rv2): tests whether extra rooms/weather add signal beyond overfit risk
    "F3_full": [TARGET] + ALL_COLS,
}
print({k: len(v) for k, v in FEAT_SETS.items()}, flush=True)
print("F3_full n feats:", len(FEAT_SETS["F3_full"]), flush=True)

# ---------- model ----------
class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TF(nn.Module):
    def __init__(self, n_feat, d_model=32, nhead=2, layers=1, ff=64, drop=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = PosEnc(d_model)
        lyr = nn.TransformerEncoderLayer(d_model, nhead, ff, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(lyr, layers)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):  # x: (B, L, F); within-window order preserved (no timestep shuffle)
        h = self.drop(self.enc(self.pos(self.proj(x))))
        return self.head(h[:, -1, :]).squeeze(-1)  # last-step pooling: strictly past-only context

def make_windows(feats_scaled, y_scaled, idx_targets, L, hist_lo=0):
    """idx_targets: target row positions; window = rows [i-L, i) -> must be >= hist_lo (past only)."""
    assert (idx_targets >= 800).sum() == 0, "TEST LEAK: targets >= 800 forbidden in stage 1"
    assert (idx_targets - L >= hist_lo).all()
    X = np.stack([feats_scaled[i-L:i] for i in idx_targets])
    y = y_scaled[idx_targets]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def run_config(feat_cols, L, d_model, nhead, layers, ff, drop, lr, batch, wd, shuffle, patience=20, max_epochs=300, verbose=False):
    set_seed()  # identical init discipline per config
    # scaler fitted on TRAIN ONLY
    scX = StandardScaler().fit(df[feat_cols].iloc[:N_TRAIN_END].values)
    scy = StandardScaler().fit(df[[TARGET]].iloc[:N_TRAIN_END].values)
    mu, sd = float(scy.mean_[0]), float(scy.scale_[0])
    Fs = scX.transform(df[feat_cols].values)
    ys = scy.transform(df[[TARGET]].values).ravel()
    tr_idx = np.arange(L, N_TRAIN_END)            # train targets (full windows inside train)
    va_idx = np.arange(N_TRAIN_END, N_VAL_END)    # 160 val targets; windows reach into train tail (past only)
    Xtr, ytr = make_windows(Fs, ys, tr_idx, L, hist_lo=0)
    Xva, yva = make_windows(Fs, ys, va_idx, L, hist_lo=0)
    tr_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=shuffle)  # shuffle=windows only
    va_dl = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)
    model = TF(len(feat_cols), d_model, nhead, layers, ff, drop)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                tot += lossf(model(xb), yb).item() * len(xb); n += len(xb)
        vl = tot / n
        if vl < best - 1e-9:
            best, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.cat([model(xb) for xb, _ in va_dl]).numpy()
    yv_true = ys[va_idx] * sd + mu
    yv_pred = pv * sd + mu
    mse = float(np.mean((yv_pred - yv_true) ** 2))
    return mse, best, ep + 1

def show(name, mse, extra=""):
    print(f"  {name}: val MSE={mse:.2f} Wh^2 {extra}", flush=True)
    return mse

print("== Stage A: feature subsets (L=24, arch L1/d32/h2/ff64/drop0.1, Adam lr1e-3, bs32, shuffled) ==", flush=True)
resA = {}
for name, cols in FEAT_SETS.items():
    mse, _, _ = run_config(cols, 24, 32, 2, 1, 64, 0.1, 1e-3, 32, 0.0, True)
    resA[name] = show(name, mse, f"(n_feat={len(cols)})")
bestA = min(resA, key=resA.get); print("-> best features:", bestA, flush=True)

print("== Stage B: lookback (winner feats, same arch/hypers) ==", flush=True)
resB = {}
for L in [12, 24, 48, 72]:
    mse, _, _ = run_config(FEAT_SETS[bestA], L, 32, 2, 1, 64, 0.1, 1e-3, 32, 0.0, True)
    resB[L] = show(f"L={L}", mse)
bestL = min(resB, key=resB.get); print("-> best lookback:", bestL, flush=True)

print("== Stage C: capacity/regularization (winner feats+lookback) ==", flush=True)
archs = {
    "C1_small_L1-d32-h2-ff64-dp01": (1, 32, 2, 64, 0.1, 0.0),
    "C2_base_L2-d32-h2-ff64-dp01": (2, 32, 2, 64, 0.1, 0.0),
    "C3_wide_L2-d64-h4-ff128-dp01": (2, 64, 4, 128, 0.1, 0.0),
    "C4_reg_L2-d32-h2-ff64-dp03-wd1e4": (2, 32, 2, 64, 0.3, 1e-4),
}
resC = {}
for name, (la, dm, nh, ff, dr, wd) in archs.items():
    mse, _, _ = run_config(FEAT_SETS[bestA], bestL, dm, nh, la, ff, dr, 1e-3, 32, wd, True)
    resC[name] = show(name, mse)
bestC = min(resC, key=resC.get); print("-> best arch:", bestC, flush=True)

print("== Stage D: batch order + LR check (winner feats/lookback/arch) ==", flush=True)
la, dm, nh, ff, dr, wd = archs[bestC]
resD = {}
for tag, sh, lr in [("shuffled_lr1e-3", True, 1e-3), ("sequential_lr1e-3", False, 1e-3),
                     ("shuffled_lr5e-4", True, 5e-4), ("shuffled_lr1e-3_bs16", True, 1e-3)]:
    bs = 16 if "bs16" in tag else 32
    mse, _, _ = run_config(FEAT_SETS[bestA], bestL, dm, nh, la, ff, dr, lr, bs, wd, sh)
    resD[tag] = show(tag, mse)
bestD = min(resD, key=resD.get); print("-> best batch-order/LR:", bestD, flush=True)

print("\n===== FROZEN (validation-only) =====", flush=True)
print(f"features={bestA} {FEAT_SETS[bestA]}", flush=True)
print(f"lookback={bestL}", flush=True)
print(f"arch={bestC} {archs[bestC]}", flush=True)
print(f"train={bestD}", flush=True)
print("val MSEs:", {**{f'A:{k}': v for k, v in resA.items()},
                      **{f'B:{k}': v for k, v in resB.items()},
                      **{f'C:{k}': v for k, v in resC.items()},
                      **{f'D:{k}': v for k, v in resD.items()}}, flush=True)
