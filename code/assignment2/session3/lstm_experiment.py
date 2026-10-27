"""LSTM sequence prediction for A2_data_release.csv — target: Appliances.
Protocol:
 - chronological order preserved, no shuffling
 - test = last 20% (rows 800-999); train/val pool = rows 0-799
 - within pool: train rows 0-639, validation rows 640-799 (chronological)
 - StandardScaler fit on TRAIN rows only, applied to val/test
 - model selection (features x lookback x architecture) on validation only
 - final selected model evaluated ONCE on test set.
"""
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
warnings.filterwarnings("ignore")

CSV = "A2_data_release.csv"
TARGET = "Appliances"
FS_UNI = ["Appliances"]
FS_MULTI = ["Appliances", "lights", "T1", "RH_1", "T_out", "RH_out"]  # 6 feats

N = 1000
TEST_START = 800          # last 200 rows = test (20%)
VAL_START = 640           # train rows 0..639, val rows 640..799
LOOKBACKS = [6, 12, 24]   # 1h, 2h, 4h at 10-min sampling
EPOCHS = 100
BATCH = 32
PATIENCE = 10

df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
assert len(df) == N and df["date"].is_monotonic_increasing
assert df.isna().sum().sum() == 0

def make_sequences(values_2d, target_idx_in_matrix, lookback):
    X, y = [], []
    for i in range(lookback, len(values_2d)):
        X.append(values_2d[i - lookback:i, :])
        y.append(values_2d[i, target_idx_in_matrix])
    return np.array(X), np.array(y)

def build_lstm(n_features, arch="1x50", lr=1e-3):
    inp = keras.Input(shape=(None, n_features))
    if arch == "1x32":
        x = layers.LSTM(32)(inp)
    elif arch == "1x50":
        x = layers.LSTM(50)(inp)
    elif arch == "2x50-drop":
        x = layers.LSTM(50, return_sequences=True)(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(50)(x)
    else:
        raise ValueError(arch)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def run_config(features, lookback, arch):
    """Train on train-rows (scalers fit there), early-stop on val; return val metrics + model."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    tgt_idx = features.index(TARGET)
    # scale: fit ONLY on train rows 0..VAL_START-1 (positional .iloc; stop is exclusive)
    sc_X = StandardScaler().fit(df.iloc[0:VAL_START][features].values)
    sc_y = StandardScaler().fit(df.iloc[0:VAL_START][[TARGET]].values)
    # sc_y must match sc_X's target column (same train column); assert to prevent silent inconsistency
    assert np.isclose(sc_X.mean_[tgt_idx], sc_y.mean_[0])
    assert np.isclose(sc_X.scale_[tgt_idx], sc_y.scale_[0])
    def scaled_features(a, b):
        """Scale rows [a, b) positionally with train-fitted params."""
        return sc_X.transform(df.iloc[a:b][features].values)
    # For sequence construction, val block needs lookback history preceding it (past only, no leakage):
    Xtr_s = scaled_features(0, VAL_START)
    Xv_raw = np.vstack([scaled_features(VAL_START - lookback, VAL_START),
                        scaled_features(VAL_START, TEST_START)])
    Xtr, ytr = make_sequences(Xtr_s, tgt_idx, lookback)
    Xv, yv = make_sequences(Xv_raw, tgt_idx, lookback)
    # Xv_raw has length lookback+160 so make_sequences already yields exactly 160
    # sequences, one per val row; keep explicit guard:
    assert len(Xv) == TEST_START - VAL_START
    model = build_lstm(len(features), arch)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    h = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=EPOCHS, batch_size=BATCH,
                  verbose=0, shuffle=False, callbacks=[es])
    pv = model.predict(Xv, verbose=0).ravel()
    # inverse to original units
    pv_orig = sc_y.inverse_transform(pv.reshape(-1, 1)).ravel()
    yv_orig = sc_y.inverse_transform(yv.reshape(-1, 1)).ravel()
    mse_orig = mean_squared_error(yv_orig, pv_orig)
    return {"val_mse_orig": mse_orig, "val_mse_scaled": mean_squared_error(yv, pv),
            "epochs": len(h.history["loss"]), "model": model, "sc_X": sc_X, "sc_y": sc_y}

print("=== Stage 1: features x lookback (arch fixed 1x50) ===")
results = []
for feats, fname in [(FS_UNI, "uni"), (FS_MULTI, "multi6")]:
    for lb in LOOKBACKS:
        r = run_config(feats, lb, "1x50")
        print(f"{fname} lb={lb}: val MSE(orig)={r['val_mse_orig']:.2f} scaled={r['val_mse_scaled']:.4f} epochs={r['epochs']}")
        results.append((fname, feats, lb, r))
res = pd.DataFrame([{"features": f, "lookback": lb, "arch": "1x50",
                     "val_mse_orig": r["val_mse_orig"], "epochs": r["epochs"]} for f, _, lb, r in results])
print(res.to_string(index=False))
best = min(results, key=lambda t: t[3]["val_mse_orig"])
best_fname, best_feats, best_lb = best[0], best[1], best[2]
print(f"--> best stage-1: {best_fname} lb={best_lb}")

print("=== Stage 2: architecture comparison at best (features, lookback) ===")
arch_results = []
for arch in ["1x32", "1x50", "2x50-drop"]:
    r = run_config(best_feats, best_lb, arch)
    print(f"{arch}: val MSE(orig)={r['val_mse_orig']:.2f} scaled={r['val_mse_scaled']:.4f} epochs={r['epochs']}")
    arch_results.append((arch, r))
best_arch, best_r = min(arch_results, key=lambda t: t[1]["val_mse_orig"])
print(f"--> SELECTED: features={best_fname} {best_feats}, lookback={best_lb}, arch={best_arch}")

# === FINAL: single test evaluation (test history may extend into val past — legitimate) ===
model, sc_X, sc_y = best_r["model"], best_r["sc_X"], best_r["sc_y"]
tgt_idx = best_feats.index(TARGET)
Xt_raw = np.vstack([sc_X.transform(df.iloc[TEST_START - best_lb:TEST_START][best_feats].values),
                    sc_X.transform(df.iloc[TEST_START:][best_feats].values)])
Xt, yt = make_sequences(Xt_raw, tgt_idx, best_lb)
assert len(Xt) == N - TEST_START  # one prediction per test row
pt = model.predict(Xt, verbose=0).ravel()
pt_orig = sc_y.inverse_transform(pt.reshape(-1, 1)).ravel()
yt_orig = sc_y.inverse_transform(yt.reshape(-1, 1)).ravel()
test_mse = mean_squared_error(yt_orig, pt_orig)
test_rmse = float(np.sqrt(test_mse)); test_mae = mean_absolute_error(yt_orig, pt_orig)
print(f"TEST (once): MSE={test_mse:.2f} (Wh^2) RMSE={test_rmse:.2f} MAE={test_mae:.2f} | scaled MSE={mean_squared_error(yt, pt):.4f}")

# naive persistence baseline on test (predict previous observed value y_{t-1})
prev_actual = df.iloc[TEST_START - 1:N - 1][TARGET].values
base_mse = mean_squared_error(yt_orig, prev_actual)
print(f"Persistence baseline test MSE={base_mse:.2f}")
# train-mean baseline (constant forecast = train mean; fit on train only, no peeking)
train_mean = float(df.iloc[0:VAL_START][TARGET].mean())
mean_mse = mean_squared_error(yt_orig, np.full_like(yt_orig, train_mean))
print(f"Train-mean baseline test MSE={mean_mse:.2f}")

test_dates = df.iloc[TEST_START:]["date"].values
plt.figure(figsize=(12, 4))
plt.plot(test_dates, yt_orig, label="Actual", linewidth=1.5)
plt.plot(test_dates, pt_orig, label="Predicted (LSTM)", linewidth=1.2)
plt.xlabel("Date"); plt.ylabel("Appliances (Wh)")
plt.title(f"LSTM predicted vs actual — test set (lb={best_lb}, {best_arch}, feats={best_fname})")
plt.legend(); plt.tight_layout()
plt.savefig("lstm_test_pred_vs_actual.png", dpi=150)
print("saved lstm_test_pred_vs_actual.png")

with open("lstm_results.txt", "w") as f:
    f.write(f"TARGET={TARGET}\nFEATURES[{best_fname}]={best_feats}\nLOOKBACK={best_lb}\nARCH={best_arch}\n")
    f.write(f"split: train 0-639 / val 640-799 / test 800-999; scaler=StandardScaler fit on train only\n")
    f.write(f"optimizer=Adam lr=1e-3 batch={BATCH} max_epochs={EPOCHS} earlystopping_patience={PATIENCE} (monitor val_loss, shuffle=False)\n")
    f.write(f"val_MSE_orig={best_r['val_mse_orig']:.4f} epochs={best_r['epochs']}\n")
    f.write(f"TEST_MSE={test_mse:.4f} TEST_RMSE={test_rmse:.4f} TEST_MAE={test_mae:.4f} TEST_scaledMSE={mean_squared_error(yt, pt):.6f}\n")
    f.write(f"PERSISTENCE_TEST_MSE={base_mse:.4f}\n")
    f.write(f"TRAINMEAN_TEST_MSE={mean_mse:.4f} (constant c=train mean {train_mean:.4f})\n")
