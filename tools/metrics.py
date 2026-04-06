import csv, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--infile", default="outputs/scores_live.csv")
args = ap.parse_args()

CSV_PATH = Path(args.infile)
print(f"📄 File: {CSV_PATH}")
ATTACK_ERR = 1.0  # heuristic: errors > 1.0 = attack

ts, idxs, err, flag, thr = [], [], [], [], []
with open(CSV_PATH, newline="") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        if len(row) < 5:
            continue
        ts.append(int(float(row[0])))
        idxs.append(int(float(row[1])))
        err.append(float(row[2]))
        flag.append(int(float(row[3])))
        t = row[4]
        thr.append(float(t) if t not in ("", "nan") else math.nan)

ts = np.array(ts)
idxs = np.array(idxs)
err  = np.array(err)
flag = np.array(flag)
thr  = np.array(thr)

# --- Ground truth ---
y_true = (err > ATTACK_ERR).astype(int)  # "attack" if error > 1.0
y_pred = flag.astype(int)                # model prediction (flag)

# --- Basic metrics ---
TP = int(((y_pred==1) & (y_true==1)).sum())
FP = int(((y_pred==1) & (y_true==0)).sum())
TN = int(((y_pred==0) & (y_true==0)).sum())
FN = int(((y_pred==0) & (y_true==1)).sum())

prec = TP / (TP+FP) if (TP+FP)>0 else 0.0
rec  = TP / (TP+FN) if (TP+FN)>0 else 0.0
f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
acc  = (TP+TN)/max(1, len(y_true))

print(f"📁 File: {CSV_PATH}")
print(f"Counts  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
print(f"Metrics acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

# --- ROC + PR helper functions ---
def roc_curve(scores, labels):
    order = np.argsort(-scores)
    s = scores[order]; y = labels[order]
    P = y.sum(); N = len(y)-P
    tps, fps = [], []
    tp = fp = 0
    last = None
    for i in range(len(y)):
        if last is None or s[i] != last:
            tps.append(tp); fps.append(fp)
            last = s[i]
        if y[i]==1: tp+=1
        else: fp+=1
    tps.append(tp); fps.append(fp)
    tpr = np.array(tps)/max(1,P)
    fpr = np.array(fps)/max(1,N)
    return fpr, tpr

def auc(x, y):
    return float(np.trapz(y, x))

def pr_curve(scores, labels):
    order = np.argsort(-scores)
    s = scores[order]; y = labels[order]
    tp = fp = 0
    precs, recs = [], []
    P = y.sum()
    for i in range(len(y)):
        if y[i]==1: tp+=1
        else: fp+=1
        precs.append(tp/max(1,tp+fp))
        recs.append(tp/max(1,P))
    return np.array(recs), np.array(precs)

# --- Compute curves + save plots ---
plots = Path("plots"); plots.mkdir(exist_ok=True)

fpr, tpr = roc_curve(err, y_true)
roc_auc = auc(fpr, tpr)
rec, precs = pr_curve(err, y_true)
pr_auc = auc(rec, precs)

plt.figure(figsize=(9,4))
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=150)

plt.figure(figsize=(9,4))
plt.plot(rec, precs, label=f'PR (AUC={pr_auc:.3f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/pr_curve.png", dpi=150)

print(f"✅ ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f}")
print("📊 Plots saved in plots/roc_curve.png and plots/pr_curve.png")
