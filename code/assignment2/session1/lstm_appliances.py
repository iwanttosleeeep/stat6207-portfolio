"""LSTM sequence prediction for A2_data_release.csv -> target: Appliances.
Protocol:
 - chronological train(0:700)/test(700:1000); NO shuffle
 - StandardScaler fit on train only
 - model selection (lookback x LSTM size) on chronological train_fit/val split inside train only
 - final refit on full train with fixed epochs, single test evaluation
"""
import os, random, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

PATH = "A2_data_release.csv"
df = pd.read_csv(PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

TARGET = "Appliances"
DROP = ["date", "rv1", "rv2"]  # rv1==rv2 exactly (duplicate noise, r~0.03 with target)
FEATURES = [c for c in df.columns if c not in DROP]

# --- chronological split: 70% train / 30% test ---
N = len(df)
N_TRAIN = 700
train_df = df.iloc[:N_TRAIN].reset_index(drop=True)
test_df  = df.iloc[N_TRAIN:].reset_index(drop=True)
print(f"N={N} train={len(train_df)} test={len(test_df)}")
print(f"train period {train_df.date.min()} -> {train_df.date.max()}")
print(f"test  period {test_df.date.min()} -> {test_df.date.max()}")

# --- normalization: fit on train only ---
scX = StandardScaler().fit(train_df[FEATURES].values)
scy = StandardScaler().fit(train_df[[TARGET]].values)
Xtr = scX.transform(train_df[FEATURES].values)
ytr = scy.transform(train_df[[TARGET]].values).ravel()
Xte = scX.transform(test_df[FEATURES].values)
yte_scaled = scy.transform(test_df[[TARGET]].values).ravel()
yte = test_df[TARGET].values

def make_seq(X, y, L):
    Xs, ys = [], []
    for i in range(L, len(X)):
        Xs.append(X[i-L:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_model(n_feat, units, dropout):
    m = keras.Sequential([
        layers.Input(shape=(None, n_feat)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m

# --- model selection on train only: chronological fit/val split ---
VAL_FRAC = 0.2  # last 20% of train sequences as validation
LOOKBACKS = [6, 12, 24]      # 1h, 2h, 4h at 10-min sampling
ARCH = [(32, 0.0), (64, 0.2)]  # (lstm units, dropout)
BATCH, MAXEP, PAT = 32, 100, 10

results = []
for L in LOOKBACKS:
    Xa, ya = make_seq(Xtr, ytr, L)
    n = len(Xa); nval = int(n*VAL_FRAC)
    Xf, yf = Xa[:n-nval], ya[:n-nval]
    Xv, yv = Xa[n-nval:], ya[n-nval:]
    yv_orig = scy.inverse_transform(yv.reshape(-1,1)).ravel()
    for units, do in ARCH:
        tf.random.set_seed(SEED)
        m = build_model(Xa.shape[2], units, do)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=PAT, restore_best_weights=True)]
        h = m.fit(Xf, yf, validation_data=(Xv, yv), epochs=MAXEP, batch_size=BATCH, verbose=0, callbacks=cb, shuffle=False)
        pv = scy.inverse_transform(m.predict(Xv, verbose=0)).ravel()
        vmse = mean_squared_error(yv_orig, pv)
        # NOTE: total epochs run includes patience overshoot; the quaity-optimal
        # epoch is argmin(val_loss). Record both; use best_epoch for final refit.
        best_epoch = int(np.argmin(h.history["val_loss"]) + 1)
        results.append(dict(lookback=L, units=units, dropout=do,
                            epochs=best_epoch, run_epochs=len(h.history["loss"]),
                            best_val_scaled=min(h.history["val_loss"]),
                            val_mse_orig=vmse))
        print(f"L={L:2d} units={units} do={do} best_epoch={best_epoch:3d} (ran {len(h.history['loss']):3d}) valMSE(orig)={vmse:.1f}")

res = pd.DataFrame(results).sort_values("val_mse_orig")
print(res.to_string(index=False))
best = res.iloc[0].to_dict()
print("BEST:", best)

# --- final refit on FULL train (fixed config+best_epoch from selection), then ONE test eval ---
# Test history correctly prepends the last L train rows so ALL 300 test points
# are predicted (previously the first L test points were dropped for history).
L, U, D = int(best["lookback"]), int(best["units"]), float(best["dropout"])
EPOCHS = int(best["epochs"])  # this is now best_epoch (argmin val_loss), not run length
XtrS, ytrS = make_seq(Xtr, ytr, L)
Xbuf = np.concatenate([Xtr[-L:], Xte], axis=0)
XteS = np.array([Xbuf[j:j+L] for j in range(len(Xte))])
yte_aligned = test_df[TARGET].values  # all 300 test targets
test_dates = test_df["date"].values
tf.random.set_seed(SEED)
final = build_model(XtrS.shape[2], U, D)
final.fit(XtrS, ytrS, epochs=EPOCHS, batch_size=BATCH, verbose=0, shuffle=False)
pred = scy.inverse_transform(final.predict(XteS, verbose=0)).ravel()

mse = mean_squared_error(yte_aligned, pred)
print(f"\nFINAL config: lookback={L} (={L*10}min), LSTM units={U}, dropout={D}, epochs={EPOCHS}, batch={BATCH}, Adam 1e-3")
print(f"TEST MSE={mse:.2f} RMSE={mse**0.5:.2f} MAE={mean_absolute_error(yte_aligned,pred):.2f} R2={r2_score(yte_aligned,pred):.3f}")
# naive baselines over the same 300 test points (computed on test only for context, not used for selection)
base_pred = np.concatenate([train_df[TARGET].values[-1:], test_df[TARGET].values[:-1]])
print(f"Baseline persistence(lag-1) MSE={mean_squared_error(yte_aligned, base_pred):.2f}")
print(f"Baseline train-mean MSE={mean_squared_error(yte_aligned, np.full_like(yte_aligned, train_df[TARGET].mean())):.2f}")
final.save("lstm_final.keras")
np.savez("lstm_test_preds.npz", date=test_dates, actual=yte_aligned, predicted=pred,
         mse=mse, lookback=L, units=U, dropout=D, epochs=EPOCHS)

# --- plots ---
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(test_dates, yte_aligned, label="actual", lw=1)
ax.plot(test_dates, pred, label="predicted", lw=1)
ax.set_title(f"Appliances: predicted vs actual (test, MSE={mse:.1f})")
ax.set_xlabel("date"); ax.set_ylabel("Appliances (Wh)"); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("pred_vs_actual_timeseries.png", dpi=150)
fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.scatter(yte_aligned, pred, s=8, alpha=0.5)
lim = [0, max(yte_aligned.max(), pred.max())*1.05]
ax2.plot(lim, lim, "r--"); ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel("actual"); ax2.set_ylabel("predicted"); ax2.set_title("Predicted vs actual (test scatter)")
ax2.set_aspect("equal", adjustable="box"); fig2.tight_layout()
fig2.savefig("pred_vs_actual_scatter.png", dpi=150)
print("saved lstm_final.keras, lstm_test_preds.npz, pred_vs_actual_timeseries.png, pred_vs_actual_scatter.png")
