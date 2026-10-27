"""LSTM 1-step-ahead prediction for A2_data_release.csv (Appliances energy).
Protocol:
- Chronological split: first 800 train, last 200 test (no shuffle of observations).
- Model selection (lookback, architecture) uses ONLY train: 640 fit / 160 val (chronological).
- StandardScaler fit on fit-portion only during selection; refit on full 800 for final
  (val block is part of train, so this is not test leakage).
- Test set transformed with train-fitted scaler and evaluated ONCE.
- Task is 1-step-ahead (10 min) using true observed history, NOT multi-step/recursive forecast.
  Windows are shuffled during gradient descent (standard for stateless LSTMs; within-window
  order is preserved); observation splits are never shuffled.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

df = pd.read_csv("A2_data_release.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

FEATURES = ["Appliances","lights","T1","RH_1","T_out","RH_out",
            "Press_mm_hg","Windspeed","Visibility","Tdewpoint"]
TARGET = "Appliances"
N = len(df); N_TRAIN = 800; N_TEST = N - N_TRAIN
N_FIT = 640; N_VAL = N_TRAIN - N_FIT  # chronological within train

print(f"N={N}, train={N_TRAIN} ({df.date.iloc[0]} -> {df.date.iloc[N_TRAIN-1]}), "
      f"test={N_TEST} ({df.date.iloc[N_TRAIN]} -> {df.date.iloc[-1]})")
print("Train-only correlations with Appliances (first 640 obs):")
print(df.iloc[:N_FIT][FEATURES].corrwith(df.iloc[:N_FIT][TARGET]).sort_values(ascending=False).to_string())

def make_seqs(arr2d, target_idx, L):
    X, y = [], []
    for i in range(L, len(arr2d)):
        X.append(arr2d[i-L:i]); y.append(arr2d[i, target_idx])
    return np.array(X), np.array(y)

def build_lstm(n_feat, units=50, nlayers=1, dropout=0.2, lr=0.001):
    # nlayers==1: single LSTM(units); nlayers==2: LSTM(units, return_seq) -> LSTM(25)
    m = keras.Sequential()
    if nlayers == 1:
        m.add(layers.LSTM(units, input_shape=(None, n_feat)))
        m.add(layers.Dropout(dropout))
    elif nlayers == 2:
        m.add(layers.LSTM(units, return_sequences=True, input_shape=(None, n_feat)))
        m.add(layers.Dropout(dropout))
        m.add(layers.LSTM(25))
        m.add(layers.Dropout(dropout))
    else:
        raise ValueError("nlayers must be 1 or 2")
    m.add(layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return m

def run_config(L, units, nlayers, seed=42, verbose=0):
    np.random.seed(seed); tf.random.set_seed(seed)
    # scale fit on FIT only
    sc = StandardScaler().fit(df.iloc[:N_FIT][FEATURES].values)
    t_idx = FEATURES.index(TARGET)
    fit_s = sc.transform(df.iloc[:N_FIT][FEATURES].values)
    val_s = sc.transform(df.iloc[N_FIT:N_TRAIN][FEATURES].values)
    # scale target separately? No: single scaler on all features; target col scaled consistently.
    # Build sequences; val sequences need lookback history crossing the boundary:
    # prepend last L rows of fit to val for continuity (scaler already fixed, no leakage of val stats).
    val_hist = np.vstack([fit_s[-L:], val_s])
    Xtr, ytr = make_seqs(fit_s, t_idx, L)
    Xv, yv = make_seqs(val_hist, t_idx, L)
    # target scale for early stopping in scaled space (monotonic w/ original)
    model = build_lstm(len(FEATURES), units, nlayers)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    h = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=100, batch_size=32,
                  callbacks=[es], verbose=verbose, shuffle=True)
    val_mse_scaled = float(np.min(h.history["val_loss"]))
    # convert val MSE to original units via target scale (linear scaling: Var(aX)=a^2 Var(X))
    s = sc.scale_[t_idx]
    return val_mse_scaled * s**2, h

print("\n--- Stage A: lookback selection (base arch 1x50, mean over seeds 42,43,44) ---")
SEEDS = [42, 43, 44]
base = {}
for L in [6, 12, 24]:
    vmses = [run_config(L, 50, 1, seed=s)[0] for s in SEEDS]
    base[L] = float(np.mean(vmses))
    print(f"L={L:2d} ({L*10} min): val MSE mean={np.mean(vmses):.1f} (seeds {['%.1f' % v for v in vmses]})")
BEST_L = min(base, key=lambda k: base[k])
print(f"Best lookback: {BEST_L}")

print(f"\n--- Stage B: architecture selection (L={BEST_L}, mean over seeds) ---")
arch = {}
for name, (u, nl) in {"1x32": (32,1), "1x50": (50,1), "2-layer 50-25": (50,2)}.items():
    vmses = [run_config(BEST_L, u, nl, seed=s)[0] for s in SEEDS]
    arch[name] = (float(np.mean(vmses)), u, nl)
    print(f"{name}: val MSE mean={np.mean(vmses):.1f} (seeds {['%.1f' % v for v in vmses]})")
BEST_ARCH = min(arch, key=lambda k: arch[k][0])
bu, bnl = arch[BEST_ARCH][1], arch[BEST_ARCH][2]
print(f"Best arch: {BEST_ARCH}")

# --- Final model: refit scaler on full 800, retrain, early stop on same val block, eval test ONCE ---
L = BEST_L
scF = StandardScaler().fit(df.iloc[:N_TRAIN][FEATURES].values)
t_idx = FEATURES.index(TARGET)
train_s = scF.transform(df.iloc[:N_TRAIN][FEATURES].values)
test_s = scF.transform(df.iloc[N_TRAIN:][FEATURES].values)
fit_s, val_hist = train_s[:N_FIT], np.vstack([train_s[N_FIT-L:N_FIT], train_s[N_FIT:N_TRAIN]])
test_hist = np.vstack([train_s[-L:], test_s])
Xfit, yfit = make_seqs(fit_s, t_idx, L)
Xval, yval = make_seqs(val_hist, t_idx, L)
Xte, yte_s = make_seqs(test_hist, t_idx, L)
np.random.seed(SEED); tf.random.set_seed(SEED)
final = build_lstm(len(FEATURES), bu, bnl)
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
final.fit(Xfit, yfit, validation_data=(Xval, yval), epochs=100, batch_size=32, callbacks=[es], verbose=0, shuffle=True)
pred_s = final.predict(Xte, verbose=0).ravel()
mu, s = scF.mean_[t_idx], scF.scale_[t_idx]
pred, actual = pred_s*s + mu, yte_s*s + mu
test_mse = float(np.mean((pred-actual)**2))
# Baselines computed on TEST once (no tuning): train-mean and naive persistence (lag-1).
train_mean = float(df.iloc[:N_TRAIN][TARGET].mean())
mse_mean = float(np.mean((actual-train_mean)**2))
persist = np.concatenate([[df.iloc[N_TRAIN-1][TARGET]], actual[:-1]])  # y_{t-1} as yhat_t
mse_persist = float(np.mean((actual-persist)**2))
arch_label = f"{'2-layer' if bnl==2 else '1x'+str(bu)}"
print(f"\nFINAL: L={L} ({L*10} min), arch={BEST_ARCH} ({arch_label} + Dropout 0.2 + Dense 1), "
      f"features={len(FEATURES)}, Adam lr=0.001, batch=32, early-stopping patience=10, seed={SEED}")
print(f"TEST MSE (original units, evaluated ONCE) = {test_mse:.2f}  RMSE={np.sqrt(test_mse):.2f}")
print(f"Baselines on same test: train-mean MSE={mse_mean:.1f}; persistence(lag-1) MSE={mse_persist:.1f}")
print(f"Test actual mean/var: {actual.mean():.1f}/{actual.var():.1f}")

# plot (Xte has one prediction per test row: test_hist length L+200 -> 200 seqs)
test_dates = df.iloc[N_TRAIN:].date.values
assert len(pred)==len(actual)==len(test_dates)==N_TEST
plt.figure(figsize=(12,4))
plt.plot(test_dates, actual, label="Actual", lw=1)
plt.plot(test_dates, pred, label="Predicted", lw=1)
plt.xlabel("Date"); plt.ylabel("Appliances (Wh per 10 min)")
plt.title(f"LSTM 1-step-ahead predicted vs actual on TEST (L={L}, {BEST_ARCH}) — Test MSE={test_mse:.1f} vs persist={mse_persist:.1f}")
plt.legend(); plt.tight_layout(); plt.xticks(rotation=20)
plt.savefig("lstm_test_pred_vs_actual.png", dpi=150)
print("Saved lstm_test_pred_vs_actual.png")
pd.DataFrame({"date": df.iloc[N_TRAIN:].date.values, "actual": actual, "predicted": pred}).to_csv("lstm_test_predictions.csv", index=False)
