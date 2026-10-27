"""Transformer experiment - STAGE 2: SINGLE test evaluation with FROZEN decisions.

Frozen (from stage_val.py, train/val only):
  features = ['Appliances'] (F1_univariate), lookback L=24,
  arch = 1 encoder layer, d_model=32, heads=2, ff=64, dropout=0.1, last-step head,
  Adam lr=1e-3, batch=32, max_epochs=300, patience=20 (val loss, best restore),
  SEQUENTIAL window presentation (shuffle=False), weight_decay=0, seed=42.

This script evaluates the test set EXACTLY ONCE. Do not re-run to tune.
"""
import math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)

# ---- frozen decisions ----
FEATS = ["Appliances"]; L = 24
D_MODEL, NHEAD, LAYERS, FF, DROP = 32, 2, 1, 64, 0.1
LR, BATCH, WD, SHUFFLE = 1e-3, 32, 0.0, False
PATIENCE, MAX_EPOCHS = 20, 300
TARGET = "Appliances"

df = pd.read_csv("A2_data_release.csv")
assert len(df) == 1000 and df.isna().sum().sum() == 0
assert (df["rv1"] == df["rv2"]).all()
y = df[TARGET].values.astype(float)
dates = pd.to_datetime(df["date"])
N_TR, N_VA = 640, 800
train_mean = float(y[:N_TR].mean())

# scaler fitted on TRAIN ONLY
scX = StandardScaler().fit(df[FEATS].iloc[:N_TR].values)
scy = StandardScaler().fit(df[[TARGET]].iloc[:N_TR].values)
mu, sd = float(scy.mean_[0]), float(scy.scale_[0])
Fs = scX.transform(df[FEATS].values)
ys = scy.transform(df[[TARGET]].values).ravel()

def windows(idxs):
    X = np.stack([Fs[i-L:i] for i in idxs])  # strictly past-only: rows [i-L, i)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(ys[idxs], dtype=torch.float32)

tr_idx = np.arange(L, N_TR)      # 616 train windows
va_idx = np.arange(N_TR, N_VA)   # 160 val targets (windows use train tail)
te_idx = np.arange(N_VA, 1000)   # 200 test targets (windows use train/val tail)
assert len(va_idx) == 160 and len(te_idx) == 200
Xtr, ytr = windows(tr_idx); Xva, yva = windows(va_idx); Xte, yte = windows(te_idx)

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
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(len(FEATS), D_MODEL)
        self.pos = PosEnc(D_MODEL)
        lyr = nn.TransformerEncoderLayer(D_MODEL, NHEAD, FF, DROP, batch_first=True)
        self.enc = nn.TransformerEncoder(lyr, LAYERS)
        self.drop = nn.Dropout(DROP)
        self.head = nn.Linear(D_MODEL, 1)
    def forward(self, x):
        return self.head(self.drop(self.enc(self.pos(self.proj(x))))[:, -1, :]).squeeze(-1)

tr_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=SHUFFLE)
va_dl = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)
model = TF()
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
lossf = nn.MSELoss()
best, best_state, bad, stop_ep = float("inf"), None, 0, MAX_EPOCHS
for ep in range(MAX_EPOCHS):
    model.train()
    for xb, yb in tr_dl:
        opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval(); tot = 0.0
    with torch.no_grad():
        for xb, yb in va_dl:
            tot += lossf(model(xb), yb).item() * len(xb)
    vl = tot / len(va_idx)
    if vl < best - 1e-9:
        best, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= PATIENCE:
            stop_ep = ep + 1; break
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pv = model(Xva).numpy()
    pt = model(Xte).numpy()
val_pred, val_true = pv * sd + mu, ys[va_idx] * sd + mu
tst_pred, tst_true = pt * sd + mu, ys[te_idx] * sd + mu

def metrics(a, p):
    e = p - a
    return float(np.mean(e**2)), float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e)))

val_mse, val_rmse, val_mae = metrics(val_true, val_pred)
tst_mse, tst_rmse, tst_mae = metrics(tst_true, tst_pred)
# baselines on TEST (original units): (a) persistence y_{t-1}; (b) train-mean constant
pers = y[te_idx - 1]
b_p = metrics(tst_true, pers)
b_m = metrics(tst_true, np.full_like(tst_true, train_mean))

print(f"seed={SEED} epochs_run={stop_ep} best_scaled_val_loss={best:.6f}")
print(f"VAL   MSE={val_mse:.2f} RMSE={val_rmse:.2f} MAE={val_mae:.2f} (n=160, original Wh units)")
print(f"TEST  MSE={tst_mse:.2f} RMSE={tst_rmse:.2f} MAE={tst_mae:.2f} (n=200, SINGLE evaluation)")
print(f"BASE persistence: MSE={b_p[0]:.2f} RMSE={b_p[1]:.2f} MAE={b_p[2]:.2f}")
print(f"BASE train-mean({train_mean:.2f}Wh): MSE={b_m[0]:.2f} RMSE={b_m[1]:.2f} MAE={b_m[2]:.2f}")

out = pd.DataFrame({"date": df["date"].iloc[te_idx].values, "actual_Wh": tst_true,
                    "pred_Wh": tst_pred, "persist_Wh": pers})
out.to_csv("transformer_test_predictions.csv", index=False)

plt.figure(figsize=(12, 4))
plt.plot(dates.iloc[te_idx], tst_true, label="actual", linewidth=1)
plt.plot(dates.iloc[te_idx], tst_pred, label="transformer pred", linewidth=1)
plt.xlabel("time (test rows 800-999)"); plt.ylabel("Appliances (Wh)")
plt.title("Transformer: test predicted vs actual (one-step-ahead)")
plt.legend(); plt.tight_layout()
plt.savefig("transformer_test_pred_vs_actual.png", dpi=150)
print("saved transformer_test_predictions.csv + transformer_test_pred_vs_actual.png")
