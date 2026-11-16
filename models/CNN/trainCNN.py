"""Training CNN for Review Classification """
#AI Disclamier: AI was used in both the development and understandment of this code


import json
import math
import numpy as np
import torch
from pathlib import Path
from collections import Counter
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from CNN import HParams,build_model,build_baseline,build_baseline_max

ROOT = Path(__file__).resolve().parents[2]   # same as your other scripts
MAX_REVIEWS, EMBED_DIM = 10, 773
NUM_CLASSES = 5          # labels 0-4 = 1-5 stars


SPLITS = {
    "train":      ("trainItems.jsonl",      "train_embeds.np"),
    "validation": ("validationItems.jsonl", "validation_embeds.np"),
    "test":       ("testItems.jsonl",        "test_embeds.np"),
}

def pad(m):
    out = np.zeros((MAX_REVIEWS,EMBED_DIM),dtype=np.float32)
    n = min(len(m),MAX_REVIEWS)#keras.layers.ZeroPadding1D(padding=MAX_REVIEWS)
    out[:n] = m[:n] #copy the first n rows of m into the first n rows of the zero array
    return out
def load_items(path):
    items = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line: #skip blank line
                items.append(json.loads(line))
    return items


def load_split(name):
    # d = np.load(f"{name}.npz")
    # return d["X"].astype("float32"), d["y"]
    X = np.load(ROOT/f"{name}_embeds.npy", mmap_mode="r")
    meta = np.load(ROOT/f"{name}_meta.npz")
    y = np.clip(np.round(meta["targets"])-1,0,4).astype(np.int64)
    assert len(X) == len(y)
    print(name,X.shape,X.dtype,sorted(Counter(y.tolist()).items()))
    return X,y

class BatchData(keras.utils.PyDataset):
    def __init__(self,X,y,batch_size,class_weight=None,shuffle=False,**kwargs):
        super().__init__(**kwargs)
        self.X, self.y = X,y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sw = None
        if class_weight is not None:
            lut = np.array([class_weight[i] for i in range(NUM_CLASSES)], dtype="float32")
            self.sw = lut[y]
        self.order = np.arange(len(y))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.y)/self.batch_size)

    def __getitem__(self,i):
        idx = self.order[i*self.batch_size: (i+1)*self.batch_size]
        idx = np.sort(idx)
        xb = self.X[idx].astype("float32")
        yb = self.y[idx]
        if self.sw is not None:
            return xb,yb,self.sw[idx]
        return xb, yb

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)

        
def make_class_weights(y,k=NUM_CLASSES):
    counts = np.bincount(y,minlength=k)
    present = counts > 0
    raw = np.zeros(k)
    raw[present] = np.sqrt(counts.sum()/counts[present])
    w = raw.copy()
    w[present] = raw[present] / np.mean(raw[present])
    w = np.minimum(w,10)
    return {i: float(w[i]) for i in range(k)}

def qwk(y_true,y_pred,k=NUM_CLASSES):
    O = np.bincount(y_true*k+y_pred,minlength=k*k).reshape(k,k).astype(float)
    idx = np.arange(k)
    W = (idx[:,None]-idx[None,:])**2/(k-1)**2
    E = np.outer(O.sum(1), O.sum(0))/O.sum()
    denom = (W*E).sum()
    return 1 - (W*O).sum() / denom if denom > 0 else 0.0

def predict_logits(model, X, chunk=4096):
    out = []
    for i in range(0, len(X), chunk):
        xb = np.asarray(X[i:i + chunk], dtype="float32")
        out.append(model.predict(xb, batch_size=256, verbose=0))
    return np.concatenate(out)

class QWKCallback(keras.callbacks.Callback):
    def __init__(self,X_val,y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
    def on_epoch_end(self, epoch, logs=None):
        pred = predict_logits(self.model,self.X_val).argmax(1)
        m = compute_metrics(self.y_val, pred)
        if logs is not None:
            logs["val_qwk"] = m["qwk"]
            logs["val_mae"] = m["mae"]
            logs["val_macro_f1"] = m["macro_f1"]


def compute_metrics(y_true, y_pred, k=NUM_CLASSES):
    O = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k).astype(float)

    # per-class F1 from the confusion matrix
    tp = np.diag(O)
    fp = O.sum(0) - tp          # predicted as class c, but wrong
    fn = O.sum(1) - tp          # truly class c, but missed
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)

    err = np.abs(y_true - y_pred)
    return {
        "qwk": qwk(y_true, y_pred, k),
        "macro_f1": f1.mean(),
        "mae": err.mean(),
        "off_by_1": (err <= 1).mean(),
        "acc": (err == 0).mean(),
        "per_class_f1": [round(float(v), 3) for v in f1],
    }

def show_metrics(name, m):
    print(f"{name:<24} QWK {m['qwk']:.3f} | macro-F1 {m['macro_f1']:.3f} | "
          f"MAE {m['mae']:.3f} | off-by-1 {m['off_by_1']:.3f} | acc {m['acc']:.3f}")
    print(f"{'':<24} per-class F1: {m['per_class_f1']}")


def evaluate_model(model, X, y):
    # logits = model.predict(X, batch_size=256, verbose=0)
    # pred = logits.argmax(1)                 # which axis holds the classes?
    # return compute_metrics(y, pred)
    pred = predict_logits(model,X).argmax(1)
    return compute_metrics(y,pred)

def run_seeds(build_fn, hp, class_weight, X_train, y_train, X_val, y_val,
              seeds=(0, 1, 2)):
    results = []
    models = []
    for seed in seeds:                       # use the argument, not a hard-coded tuple
        #seed FIRST, so weight init is seeded
        keras.utils.set_random_seed(seed)

        # fresh model every seed
        model = build_fn(hp)

        #fresh optimizer via compile
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=hp.lr, weight_decay=.02),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        #fresh callbacks (QWK before EarlyStopping)
        callbacks = [
            QWKCallback(X_val, y_val),
            keras.callbacks.EarlyStopping(monitor="val_qwk", mode="max",
                                          patience=3, restore_best_weights=True),
        ]

    
        train_ds = BatchData(X_train, y_train, hp.batch_size,
                     class_weight=class_weight, shuffle=True)   # inside the loop, fresh per seed
        model.fit(train_ds, epochs=hp.epochs, callbacks=callbacks, verbose=0)
        m = evaluate_model(model, X_val, y_val)       # validation arrays
        show_metrics(f"seed {seed} (val)", m)
        results.append(m)                         # keep full dicts, not just QWK
        models.append(model)

    return results,models

def summarize(name, results):
    for key in ("qwk", "macro_f1", "mae", "off_by_1", "acc"):
        vals = [r[key] for r in results]
        print(f"{name} {key:<9} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

if __name__ == "__main__":
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")
    hp = HParams()
    w = make_class_weights(y_train)
    print("class weights:", w)

    # train 3 seeds once (validation is used for early stopping)
    val_results, models = run_seeds(build_model, hp, w, X_train, y_train, X_val, y_val)
    summarize("full (val)", val_results)

    #evaluate those same trained models on test, once
    majority = np.bincount(y_train).argmax()
    show_metrics("always-majority (test)",
                 compute_metrics(y_test, np.full_like(y_test, majority)))

    test_results = []
    for s, mdl in enumerate(models):
        m = evaluate_model(mdl, X_test, y_test)
        show_metrics(f"full seed {s} (test)", m)
        test_results.append(m)
    summarize("full (test)", test_results)
