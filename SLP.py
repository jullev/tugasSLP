import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Paraemeters Setting
LearningRate = 0.1
InitialWeight = 0.5
Epochs = 5
OUTPUT_DIR = "output"
Data =  pd.read_csv("iris_data.csv")

FEATURES = ["x1", "x2", "x3", "x4"]
WEIGHTS = ["bias", "teta1", "teta2", "teta3", "teta4"]
LABELS = {"Iris-setosa": 0, "Iris-versicolor": 1}
print (Data)

def loadData():
    # Load the dataset
    data = pd.read_csv("iris_data.csv")
    # Map the labels to numerical values
    data['target'] = data['species'].map(LABELS)
    return data

def splitData(df, train_per_class=40):
    groups = df.groupby('target', sort=True)
    train = pd.concat([g.iloc[:train_per_class] for _, g in groups], ignore_index=True)
    val = pd.concat([g.iloc[train_per_class:] for _, g in groups], ignore_index=True)
    return train, val

def dotProduct(weights, features):
    teta = weights[["teta1", "teta2", "teta3", "teta4"]]
    return weights["bias"] + sum(t * f for t, f in zip(teta, features))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict (gz):
    return 1 if gz >= 0.5 else 0

def forward(w, x, target):
    """Hitung z, g(z), prediksi, error, dan SSE untuk satu data."""
    z = dotProduct(w, x)
    gz = sigmoid(z)
    pred = predict(gz)
    error = gz - target
    sse = error ** 2
    return z, gz, pred, error, sse

def gradients(gz,target,x):
    """Series gradien: dbias, dteta1..dteta4."""
    dbias = 2 * (gz - target) * (1 - gz) * gz
    return pd.Series([dbias, *(dbias * x.values)], index=WEIGHTS)
#training methode
def train_one_epoch(train, w, lr, epoch):
    rows = []
    for no, row in enumerate(train.itertuples(index=False), start=1):
        x = pd.Series([getattr(row, f) for f in FEATURES], index=FEATURES)
        target = row.target
 
        z, gz, pred, error, sse = forward(w, x, target)
        grad = gradients(gz, target, x)
 
        rows.append({"epoch": epoch, "no": no, **x.to_dict(), "target": target,
                    **w.to_dict(), "z": z, "g(z)": gz, "prediksi": pred,
                    "error": error, "SSE": sse,
                    **grad.add_prefix("d").to_dict()})
 
        # update bobot untuk data berikutnya
        w = w - lr * grad
 
    log = pd.DataFrame(rows)
    mse = log["SSE"].mean()
    acc = (log["prediksi"] == log["target"]).mean()
    return w, log, mse, acc
#validasi
def validate(val, w, epoch):
    rows = []
    for no, row in enumerate(val.itertuples(index=False), start=1):
        x = pd.Series([getattr(row, f) for f in FEATURES], index=FEATURES)
        z, gz, pred, error, sse = forward(w, x, row.target)
        rows.append({"epoch": epoch, "no": no, **x.to_dict(), "target": row.target,
                    **w.to_dict(), "z": z, "g(z)": gz, "prediksi": pred,
                    "error": error, "SSE": sse})
 
    log = pd.DataFrame(rows)
    mse = log["SSE"].mean()
    acc = (log["prediksi"] == log["target"]).mean()
    return log, mse, acc

def run(epochs=Epochs, lr=LearningRate, w0=InitialWeight):
    df = loadData()
    train, val = splitData(df)
 
    w = pd.Series(w0, index=WEIGHTS, dtype=float)
 
    train_logs, val_logs, history = [], [], []
    for epoch in range(1, epochs + 1):
        w, tr_log, mse_tr, acc_tr = train_one_epoch(train, w, lr, epoch)
        # validasi pakai bobot akhir epoch ini
        va_log, mse_val, acc_val = validate(val, w, epoch)
 
        train_logs.append(tr_log)
        val_logs.append(va_log)
        history.append({"epoch": epoch, "MSE_train": mse_tr, "MSE_val": mse_val,
                       "acc_train": acc_tr, "acc_val": acc_val})
 
    return (pd.concat(train_logs, ignore_index=True),
            pd.concat(val_logs, ignore_index=True),
            pd.DataFrame(history),
            w)
 
 
def plot_history(history, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib belum terpasang -> grafik dilewati (pip install matplotlib)")
        return
 
    h = history.set_index("epoch")
 
    ax = h[["MSE_train", "MSE_val"]].plot(marker="o", figsize=(7, 4.5),
                                          title="Loss / Error (MSE) per Epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_xticks(h.index)
    ax.legend(["Training", "Validasi"]); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_chart.png"), dpi=150)
 
    ax = (h[["acc_train", "acc_val"]] * 100).plot(marker="o", figsize=(7, 4.5),
                                                  title="Akurasi per Epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Akurasi (%)"); ax.set_xticks(h.index)
    ax.set_ylim(0, 105); ax.legend(["Training", "Validasi"]); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "accuracy_chart.png"), dpi=150)
 
    plt.close("all")

def main():
    train_log, val_log, history, w = run()
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_log.to_csv(os.path.join(OUTPUT_DIR, "training_detail.csv"), index=False)
    val_log.to_csv(os.path.join(OUTPUT_DIR, "validation_detail.csv"), index=False)
    history.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
 
    # contoh perhitungan data pertama (bisa dicocokkan dengan baris 5 di Excel)
    print("Contoh epoch 1, data no 1:")
    print(train_log.loc[0, ["z", "g(z)", "prediksi", "error", "SSE"]].to_string(), "\n")
 
    pd.set_option("display.float_format", "{:.6f}".format)
    print("Ringkasan per epoch:")
    print(history.to_string(index=False), "\n")
 
    print("Bobot akhir:")
    print(w.to_string())
    print(f"\nFile hasil disimpan di folder: {OUTPUT_DIR}")
 
    plot_history(history, OUTPUT_DIR)
 
 
if __name__ == "__main__":
    main()
