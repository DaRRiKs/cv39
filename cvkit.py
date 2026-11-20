import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import __version__ as torchvision_version
from torchvision.transforms.functional import gaussian_blur
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import canny, hog
from skimage.filters import sobel, threshold_otsu
from skimage.morphology import closing, disk, opening, skeletonize
from skimage.transform import resize

SEED = 1005
ROOT = Path("results")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def event(path, kind, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind, **values}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def setup(threads=2):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def environment():
    names = ["torch", "torchvision", "numpy", "scikit-learn", "scikit-image", "matplotlib", "scipy", "pillow"]
    return {"python": sys.version, "platform": platform.platform(), "cpu": cpu_name(),
            "device": "cpu", "precision": "float32", "seed": SEED,
            "threads": torch.get_num_threads(), "logical_cpus": os.cpu_count(),
            "torch_runtime": str(torch.__version__), "torchvision_runtime": torchvision_version,
            "cuda_runtime": torch.version.cuda,
            "versions": {n: importlib.metadata.version(n) for n in names}}


def record_command():
    event(ROOT / "commands.jsonl", "command", argv=sys.argv, executable=sys.executable)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_base(name, train=True):
    cls = {"mnist": datasets.MNIST, "kmnist": datasets.KMNIST}[name]
    return cls("data", train=train, download=True)


def split_data(name):
    base = load_base(name)
    folder = ROOT / "data"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_split.npz"
    labels = base.targets.numpy()
    if not path.exists():
        train, val = train_test_split(np.arange(len(base)), test_size=0.1, random_state=SEED, stratify=labels)
        np.savez(path, train=train, val=val, labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(), seed=SEED)
    with np.load(path) as saved:
        train, val = saved["train"], saved["val"]
        assert str(saved["labels_sha256"]) == hashlib.sha256(labels.tobytes()).hexdigest()
        assert int(saved["seed"]) == SEED
    assert len(train) == 54000 and len(val) == 6000
    assert np.array_equal(np.sort(np.concatenate([train, val])), np.arange(len(base)))
    return base, train, val


def normalization(base=None, train=None):
    path = ROOT / "data" / "normalization.json"
    if path.exists():
        norm = read_json(path)
        assert norm["split_sha256"] == file_hash(ROOT / "data" / "mnist_split.npz")
        return norm
    if base is None:
        base, train, _ = split_data("mnist")
    total, squares, count = 0.0, 0.0, 0
    for ids in np.array_split(train, 54):
        values = base.data[ids].numpy().astype(np.float64) / 255
        total += float(values.sum())
        squares += float((values ** 2).sum())
        count += values.size
    mean = total / count
    norm = {"mean": mean, "std": math.sqrt(squares / count - mean ** 2),
            "source": "MNIST train only", "pixels": count, "ddof": 0,
            "split_sha256": file_hash(ROOT / "data" / "mnist_split.npz")}
    save_json(path, norm)
    return norm


class Images(Dataset):
    def __init__(self, base, indices):
        self.images = base.data
        self.labels = base.targets
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        return self.images[index].unsqueeze(0).float() / 255, self.labels[index], index


def loader(base, ids, batch_size=128, shuffle=False, generator=None):
    if generator is None:
        generator = torch.Generator().manual_seed(SEED)
    return DataLoader(Images(base, ids), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def normalize(x, norm):
    return (x - norm["mean"]) / norm["std"]


def rotate_batch(x, generator):
    angle = (torch.rand(len(x), generator=generator) * 20 - 10) * math.pi / 180
    theta = torch.zeros(len(x), 2, 3)
    theta[:, 0, 0] = angle.cos()
    theta[:, 0, 1] = -angle.sin()
    theta[:, 1, 0] = angle.sin()
    theta[:, 1, 1] = angle.cos()
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0, 1)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


def describe_model(model, folder):
    rows = []
    x = torch.zeros(1, 1, 28, 28)
    with torch.no_grad():
        for name, layer in model.named_children():
            before = list(x.shape)
            x = layer(x)
            rows.append({"layer": name, "operation": str(layer), "input": str(before), "output": str(list(x.shape)),
                         "parameters": sum(p.numel() for p in layer.parameters())})
    write_csv(folder / "architecture.csv", rows)
    (folder / "architecture.txt").write_text(str(model), encoding="utf-8")
    info = {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "initial_weights_sha256": state_hash(model)}
    save_json(folder / "model_info.json", info)
    print(model, flush=True)
    print(json.dumps(info), flush=True)
    for row in rows:
        print(row["layer"], row["input"], "->", row["output"], flush=True)
    return info


def rng_state(generators):
    return {"random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "generators": {k: g.get_state() for k, g in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name])


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = LeNet5()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict(model, data_loader, norm, corruption="clean"):
    model.eval()
    scores, targets, indices = [], [], []
    total_loss = 0.0
    noise = torch.Generator().manual_seed(SEED)
    with torch.inference_mode():
        for x, y, ids in data_loader:
            if corruption == "noise":
                x = (x + 0.05 * torch.randn(x.shape, generator=noise)).clamp(0, 1)
            elif corruption == "blur":
                x = gaussian_blur(x, [5, 5], [1.0, 1.0]).clamp(0, 1)
            elif corruption in {"brightness07", "brightness13"}:
                x = (x * (0.7 if corruption == "brightness07" else 1.3)).clamp(0, 1)
            elif corruption != "clean":
                raise ValueError(corruption)
            logits = model(normalize(x, norm))
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            scores.append(logits.softmax(1).numpy())
            targets.append(y.numpy())
            indices.append(ids.numpy())
    probs, y, ids = np.concatenate(scores), np.concatenate(targets), np.concatenate(indices)
    return {"probabilities": probs, "targets": y, "indices": ids,
            "loss": total_loss / len(y), "accuracy": float((probs.argmax(1) == y).mean())}


def save_metrics(folder, y, probs, indices):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1, kind="stable")[:, :3]
    report = classification_report(y, pred, labels=list(range(10)), output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(10)))
    normalized = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    metrics = {"n": len(y), "accuracy": accuracy_score(y, pred),
               "top3_accuracy": float((top3 == y[:, None]).any(1).mean()),
               "macro_f1": f1_score(y, pred, labels=list(range(10)), average="macro", zero_division=0),
               "errors": int((y != pred).sum()), "top3_tie_rule": "descending probability, ascending class index"}
    save_json(folder / "classification_report.json", report)
    write_csv(folder / "per_class.csv", [{"class": c, **report[str(c)]} for c in range(10)])
    write_csv(folder / "predictions.csv", [{"index": int(i), "true": int(t), "predicted": int(p),
              "confidence": float(v[p])} for i, t, p, v in zip(indices, y, pred, probs)])
    np.savez_compressed(folder / "predictions.npz", indices=indices, targets=y, probabilities=probs)
    np.savetxt(folder / "confusion_counts.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(folder / "confusion_normalized.csv", normalized, fmt="%.9f", delimiter=",")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{normalized[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if normalized[i,j] > 0.5 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Предсказанный класс (столбцы)",
           ylabel="Истинный класс (строки)", title="Confusion matrix: нормировка по строкам")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(folder / "confusion.png", dpi=150)
    plt.close(fig)
    save_json(folder / "metrics.json", metrics)
    return metrics


def eda(name, include_test=False):
    base, train, val = split_data(name)
    folder = ROOT / "eda" / name
    folder.mkdir(parents=True, exist_ok=True)
    splits = {"train": (base, train), "val": (base, val)}
    if include_test:
        test = load_base(name, False)
        splits["test"] = (test, np.arange(len(test)))
    rows, info = [], {}
    for split, (ds, ids) in splits.items():
        counts = np.bincount(ds.targets[ids].numpy(), minlength=10)
        values = ds.data[ids]
        info[split] = {"n": len(ids), "shape": list(values.shape[1:]), "channels": 1,
                       "dtype": str(values.dtype), "range_uint8": [int(values.min()), int(values.max())],
                       "range_float": [float(values.min()) / 255, float(values.max()) / 255]}
        rows.extend({"split": split, "class": c, "count": int(counts[c])} for c in range(10))
    write_csv(folder / "class_counts.csv", rows)
    save_json(folder / "summary.json", info)
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 3))
    for ax, split in zip(np.atleast_1d(axes), splits):
        ax.bar(range(10), [r["count"] for r in rows if r["split"] == split])
        ax.set(title=f"{name.upper()} {split}", xlabel="Класс", ylabel="Изображений", xticks=range(10))
    fig.tight_layout()
    fig.savefig(folder / "class_hist.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(10, 3, figsize=(5, 14))
    selected = []
    for c in range(10):
        ids = train[base.targets[train].numpy() == c][:3]
        for ax, i in zip(axes[c], ids):
            ax.imshow(base.data[i], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"class={c}, index={i}", fontsize=8)
            ax.axis("off")
            selected.append({"class": c, "index": int(i)})
    fig.tight_layout()
    fig.savefig(folder / "contact_sheet.png", dpi=160)
    plt.close(fig)
    write_csv(folder / "contact_indices.csv", selected)
    if name == "mnist":
        normalization(base, train)
    return info


def processing_sheet():
    base, train, _ = split_data("mnist")
    folder = ROOT / "classical"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 8, figsize=(17, 7))
    meta = []
    for row, c in enumerate([0, 4, 8]):
        i = int(train[base.targets[train].numpy() == c][0])
        x = base.data[i].numpy().astype(np.float32) / 255
        threshold = float(threshold_otsu(x))
        mask = x > threshold
        ops = [(x, "Исходное"), (x, "Grayscale: 1 канал"), (sobel(x), "Sobel"),
               (canny(x, sigma=1), "Canny, σ=1"), (mask, f"Otsu, t={threshold:.3f}"),
               (opening(mask, disk(2)), "Opening, disk(2)"),
               (closing(mask, disk(2)), "Closing, disk(2)"), (skeletonize(mask), "Skeletonize(Otsu)")]
        for ax, (value, title) in zip(axes[row], ops):
            ax.imshow(value, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"Класс {c}, index={i}")
        meta.append({"index": i, "class": c, "otsu_threshold": threshold})
    fig.tight_layout()
    fig.savefig(folder / "processing.png", dpi=180)
    plt.close(fig)
    save_json(folder / "processing.json", meta)


def features(images, kind):
    rows = []
    for x in images:
        x = np.asarray(x, dtype=np.float32) / 255
        if kind == "raw":
            row = resize(x, (16, 16), order=1, anti_aliasing=True, preserve_range=True).reshape(-1)
        elif kind == "hog":
            row = hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
        else:
            raise ValueError(kind)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def classical_fit(kind):
    base, train, val = split_data("mnist")
    folder = ROOT / "classical" / kind
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "model.pkl").exists():
        raise FileExistsError(f"Already trained: {folder}")
    start = time.perf_counter()
    x_train = features(base.data[train].numpy(), kind)
    extract_train = time.perf_counter() - start
    start = time.perf_counter()
    x_val = features(base.data[val].numpy(), kind)
    extract_val = time.perf_counter() - start
    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, metric="euclidean", algorithm="brute", n_jobs=1))
    start = time.perf_counter()
    pipeline.fit(x_train, base.targets[train].numpy())
    fit_seconds = time.perf_counter() - start
    with (folder / "model.pkl").open("wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    start = time.perf_counter()
    probs = pipeline.predict_proba(x_val)
    predict_seconds = time.perf_counter() - start
    metrics = save_metrics(folder / "val", base.targets[val].numpy(), probs, val)
    knn = pipeline.named_steps["kneighborsclassifier"]
    scaler = pipeline.named_steps["standardscaler"]
    info = {"kind": kind, "train_n": len(train), "val_n": len(val), "feature_dim": int(x_train.shape[1]),
            "extract_train_seconds": extract_train, "extract_val_seconds": extract_val,
            "fit_seconds": fit_seconds, "train_total_seconds": extract_train + fit_seconds,
            "val_predict_seconds": predict_seconds, "feature_dtype": str(x_train.dtype),
            "stored_examples_bytes": knn._fit_X.nbytes, "stored_labels_bytes": knn._y.nbytes,
            "scaler_arrays_bytes": sum(a.nbytes for a in [scaler.mean_, scaler.scale_, scaler.var_]),
            "model_file_bytes": (folder / "model.pkl").stat().st_size,
            "split_sha256": file_hash(ROOT / "data" / "mnist_split.npz"),
            "pipeline": str(pipeline), "all_training_images_used": True}
    save_json(folder / "info.json", info)
    print(kind, info, metrics, flush=True)
    return metrics


def plot_training():
    folder = ROOT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    summary = []
    for name in ["none", "rotate10"]:
        rows = read_csv(ROOT / name / "log.csv")
        epoch = [int(r["epoch"]) for r in rows]
        for j, metric in enumerate(["loss", "accuracy"]):
            for split, style in [("train", "--"), ("val", "-")]:
                axes[j].plot(epoch, [float(r[f"{split}_{metric}"]) for r in rows], style, label=f"{name} {split}")
        axes[2].plot(epoch, [float(r["lr"]) for r in rows], label=name)
        best = max(rows, key=lambda r: float(r["val_accuracy"]))
        summary.append({"augmentation": name, "best_epoch": int(best["epoch"]),
                        "best_val_accuracy": float(best["val_accuracy"]), "last_val_accuracy": float(rows[-1]["val_accuracy"]),
                        "epoch_seconds_sum": sum(float(r["epoch_seconds"]) for r in rows),
                        "train_seconds_sum": sum(float(r["train_seconds"]) for r in rows)})
    for ax, title in zip(axes, ["Cross entropy", "Accuracy", "Learning rate"]):
        ax.set(xlabel="Эпоха", title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "ablation.png", dpi=160)
    plt.close(fig)
    write_csv(folder / "ablation.csv", summary)
    save_json(folder / "ablation_delta.json", {"rotate10_minus_none_best_pp":
              100 * (summary[1]["best_val_accuracy"] - summary[0]["best_val_accuracy"]),
              "rotate10_minus_none_last_pp": 100 * (summary[1]["last_val_accuracy"] - summary[0]["last_val_accuracy"])})


def plot_transfer():
    rows = read_csv(ROOT / "kmnist" / "domain_log.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = [int(r["epoch"]) for r in rows]
    for key, label in [("mnist_accuracy", "MNIST val: сохранение"), ("kmnist_accuracy", "KMNIST val: адаптация")]:
        ax.plot(epochs, [100 * float(r[key]) for r in rows], "o-", label=label)
    ax.set(xlabel="Эпоха KMNIST (0 — до дообучения)", ylabel="Accuracy, %", xticks=epochs)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "kmnist" / "adaptation.png", dpi=160)
    plt.close(fig)


def all_errors(base, result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    y, probs, ids = result["targets"], result["probabilities"], result["indices"]
    pred = probs.argmax(1)
    wrong = np.flatnonzero(y != pred)
    wrong = wrong[np.argsort(ids[wrong])]
    rows = [{"index": int(ids[i]), "true": int(y[i]), "predicted": int(pred[i]),
             "confidence": float(probs[i, pred[i]])} for i in wrong]
    write_csv(folder / "errors.csv", rows, ["index", "true", "predicted", "confidence"])
    gallery = []
    for page, start in enumerate(range(0, len(wrong), 30), 1):
        subset = wrong[start:start + 30]
        fig, axes = plt.subplots(math.ceil(len(subset) / 6), 6, figsize=(13, 2.1 * math.ceil(len(subset) / 6)), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, subset):
            ax.imshow(base.data[ids[i]], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"#{ids[i]}: {y[i]} → {pred[i]}\np={probs[i,pred[i]]:.3f}", fontsize=10)
        fig.tight_layout()
        name = f"errors_{page:02d}.png"
        fig.savefig(folder / name, dpi=180)
        plt.close(fig)
        gallery.append({"file": name, "indices": [int(ids[i]) for i in subset]})
    save_json(folder / "gallery_index.json", gallery)
    cm = confusion_matrix(y, pred, labels=range(10))
    pairs = [(int(cm[a, b]), a, b) for a in range(10) for b in range(10) if a != b and cm[a, b] > 0]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    write_csv(folder / "top_confusions.csv", [{"true": a, "predicted": b, "count": n,
              "fraction_of_true_class": n / int(cm[a].sum())} for n, a, b in pairs[:5]],
              ["true", "predicted", "count", "fraction_of_true_class"])
    return rows


def gradcam(model, x, target):
    activation = []
    hook = model.conv2.register_forward_hook(lambda module, inputs, output: activation.append(output))
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        grad = torch.autograd.grad(logits[0, target], activation[0])[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation[0]).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-12)
        return cam.detach().numpy()
    finally:
        hook.remove()


def gradcam_examples(model, base, result, norm, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    y, probs, ids = result["targets"], result["probabilities"], result["indices"]
    correct = probs.argmax(1) == y
    meta = []
    for name, mask in [("correct", correct), ("wrong", ~correct)]:
        chosen = np.flatnonzero(mask)
        chosen = chosen[np.argsort(ids[chosen])][:5]
        if len(chosen) == 0:
            continue
        fig, axes = plt.subplots(len(chosen), 3, figsize=(8, 2.5 * len(chosen)), squeeze=False)
        for row, i in enumerate(chosen):
            index, target = int(ids[i]), int(probs[i].argmax())
            original = base.data[index].numpy().astype(np.float32) / 255
            x = normalize(torch.from_numpy(original)[None, None], norm)
            cam = gradcam(model, x, target)
            np.save(folder / f"cam_{index}.npy", cam)
            axes[row, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
            axes[row, 2].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.45)
            axes[row, 0].set_title(f"#{index}: true={y[i]}, pred={target}", fontsize=10)
            axes[row, 1].set_title(f"Grad-CAM target={target}", fontsize=10)
            axes[row, 2].set_title("Наложение", fontsize=10)
            for ax in axes[row]:
                ax.axis("off")
            meta.append({"index": index, "true": int(y[i]), "predicted": target, "target": target,
                         "group": name, "confidence": float(probs[i,target]), "cam_max": float(cam.max())})
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=180)
        plt.close(fig)
    write_csv(folder / "examples.csv", meta)


def robustness(model, val_loader, norm, clean, folder):
    rows = [{"corruption": "clean", "accuracy": clean, "drop_pp": 0.0}]
    for kind in ["noise", "blur", "brightness07", "brightness13"]:
        result = predict(model, val_loader, norm, kind)
        rows.append({"corruption": kind, "accuracy": result["accuracy"], "drop_pp": 100 * (clean - result["accuracy"])})
    write_csv(folder / "robustness.csv", rows)
    save_json(folder / "robustness_config.json", {"noise_sigma": 0.05, "noise_seed": SEED,
              "blur_kernel": [5, 5], "blur_sigma": [1.0, 1.0], "blur_padding": "reflect",
              "brightness_factors": [0.7, 1.3], "range": [0, 1], "before_normalization": True,
              "noise_order": "fixed validation order, same batch size 128"})
    return rows


def latency(model, base, val, norm, folder):
    x = normalize(base.data[val[:120]].float().unsqueeze(1) / 255, norm)
    images = [t.unsqueeze(0).contiguous() for t in x]
    model.cpu().eval()
    timings = []
    with torch.inference_mode():
        for t in images[:20]:
            model(t)
        for index, t in zip(val[20:120], images[20:120]):
            start = time.perf_counter_ns()
            model(t)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            timings.append({"index": int(index), "milliseconds": elapsed})
    write_csv(folder / "latency_samples.csv", timings)
    mean = float(np.mean([r["milliseconds"] for r in timings]))
    info = {"mean_ms": mean, "fps": 1000 / mean, "warmup": 20, "measured": 100,
            "batch_size": 1, "device": "cpu", "precision": "float32", "cpu": cpu_name(),
            "threads": torch.get_num_threads(), "scope": "model forward only; excludes loading, normalization, softmax and file I/O"}
    save_json(folder / "latency.json", info)
    return info


def markdown_table(headers, rows):
    if not rows:
        return "Результаты не получены.\n"
    body = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        body.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |")
    return "\n".join(body) + "\n"


def write_report():
    def js(path):
        path = ROOT / path
        return read_json(path) if path.exists() else None

    def rows(path):
        path = ROOT / path
        return read_csv(path) if path.exists() else []

    def figure(path, title):
        return f"![{title}](results/{path})\n" if (ROOT / path).exists() else f"{title}: не получено.\n"

    def pct(value):
        return f"{100 * float(value):.3f}"

    def metrics_table(items):
        data = []
        for name, path in items:
            metric = js(path)
            if metric:
                data.append([name, metric["n"], pct(metric["accuracy"]), pct(metric["top3_accuracy"]), f"{metric['macro_f1']:.5f}", metric["errors"]])
        return markdown_table(["Модель / сплит", "N", "Accuracy, %", "Top-3, %", "Macro-F1", "Ошибок"], data)

    env = js("environment.json") or environment()
    main = "evaluation/rotate10/mnist"
    model_info = js("model/model_info.json")
    complete = bool(js("final_test/completed.json"))
    text = ["# Индивидуальное задание №5. MNIST + LeNet-5\n",
            "Студент: __________  \nГруппа: __________  ",
            f"Устройство: {env['cpu']}, CPU, float32, {env['threads']} потока PyTorch.  ",
            f"PyTorch: {env.get('torch_runtime', env['versions']['torch'])}; seed: 1005.  ",
            "Код: [train.py](train.py), [evaluate.py](evaluate.py), [cvkit.py](cvkit.py).\n",
            "## 1. Цель и индивидуальный вопрос\n",
            "Цель — сравнить raw pixels+kNN, HOG+kNN и учебную LeNet-5 на одинаковых сплитах MNIST, проверить повороты ±10°, "
            "изучить ошибки и адаптацию к KMNIST. Индивидуальный вопрос: какое качество достигнуто около ориентира 99% и что мешает распознать оставшиеся изображения? "
            "99% — ориентир задания, а не заранее заданный ответ. Ансамбль и дополнительные эксперименты не выполняются.\n"]
    if not complete:
        text.append("**Статус: экспериментальная работа не завершена.** Финальная оценка test не выполнена. "
                    "Пустые результаты ниже означают отсутствие измерений, а не нулевое качество.\n")
    availability = js("data_availability.json")
    if availability:
        for item in availability:
            if not item["available"]:
                text.append(f"Загрузка {item['dataset'].upper()} через torchvision завершилась ошибкой `{item['exception']}`:\n\n```text\n{item['message'].strip()}\n```\n")
    text.append("Состояние среды: [environment.json](results/environment.json); команды: [commands.jsonl](results/commands.jsonl); "
                "зафиксированный протокол: [protocol.json](results/protocol.json); полные версии: [pip-freeze.txt](results/pip-freeze.txt). "
                "Материалы задания и лекции прочитаны; отсутствующие starter templates не используются.\n")
    text += ["## 2. Данные, EDA и сплиты\n",
             "Используются `torchvision.datasets.MNIST` и `torchvision.datasets.KMNIST`: вход 1×28×28, десять классов. "
             "Плановый размер каждого train/val — 54 000/6 000 из официальных 60 000; официальный test содержит 10 000. "
             "Это спецификация датасетов; фактические размеры после загрузки приведены в таблице ниже. "
             "Стратификация: `train_test_split(test_size=0.1, stratify=labels, random_state=1005)`. "
             "Индексы сохраняются отдельно для каждого датасета; пересечений нет. Метрики test вычисляются только финальной командой. "
             "Численности классов test добавляются в EDA в этом финальном этапе, изображения test для промежуточного анализа не используются. "
             "[MNIST API](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.MNIST.html), "
             "[описание KMNIST](https://github.com/rois-codh/kmnist).\n"]
    for domain in ["mnist", "kmnist"]:
        info = js(f"eda/{domain}/summary.json")
        text.append(f"**{domain.upper()}.**")
        text.append(markdown_table(["Сплит", "Количество", "Размер", "Каналы", "uint8", "float"],
                    [[key, v["n"], str(v["shape"]), v["channels"], str(v["range_uint8"]), str(v["range_float"])] for key, v in (info or {}).items()]))
        counts = rows(f"eda/{domain}/class_counts.csv")
        text.append(markdown_table(["Класс", "Train", "Val", "Test"],
                    [[c] + [next((r["count"] for r in counts if int(r["class"]) == c and r["split"] == s), "—")
                    for s in ["train", "val", "test"]] for c in range(10)] if counts else []))
        text.append(figure(f"eda/{domain}/class_hist.png", f"Распределение классов {domain.upper()}"))
        text.append(figure(f"eda/{domain}/contact_sheet.png", f"По 3 train-изображения каждого класса {domain.upper()}"))
    norm = js("data/normalization.json")
    text.append(f"Train mean={norm['mean']:.9f}, std={norm['std']:.9f}; ddof=0.\n" if norm else "Train mean/std не вычислены: данные недоступны.\n")
    text.append("Нормализация вычисляется только по MNIST train и сохраняется в `results/data/normalization.json`. "
                "Она фиксируется для MNIST и KMNIST на всех стадиях, включая эпоху 0. Нельзя пересчитывать её на val/test. "
                "До нормализации пиксели переводятся из uint8 в float32 / 255.\n")
    text += ["## 3. Классические методы и baseline\n", figure("classical/processing.png", "Лист классической обработки"),
             "Для листа берутся первые train-примеры классов 0, 4 и 8. MNIST уже одноканальный, поэтому исходное изображение и grayscale совпадают. "
             "Sobel выделяет силу локальных перепадов яркости, Canny с σ=1 — тонкие границы штрихов. Otsu отделяет светлые штрихи от тёмного фона. "
             "Opening с disk(2) удаляет тонкие выступы и малые компоненты; closing с disk(2) заполняет небольшие разрывы. "
             "Скелетизация сводит бинарные штрихи к тонким осевым линиям. Все три операции применяются отдельно к исходной Otsu-маске, "
             "а не к grayscale и не друг к другу. Какие детали реально исчезли, можно установить только по полученному листу.\n",
             "Raw: resize 28×28 → 16×16, bilinear с anti-aliasing, диапазон [0,1], разворачивание в 256 признаков. "
             "HOG: исходные 28×28, 9 ориентаций, ячейки 8×8, блоки 2×2, L2-Hys; получается 144 признака "
             "(2×2 позиции блока × 2×2 ячейки × 9). Крайние неполные ячейки HOG не входят в дескриптор. "
             "Отдельные pipelines: StandardScaler → kNN(k=5, Euclidean, uniform). Все 54 000 train-примеров сохраняются в kNN, сокращения нет. "
             "Fit scaler и kNN — только на train. На val/test используются transform и predict_proba.\n"]
    data = []
    for kind in ["raw", "hog"]:
        info = js(f"classical/{kind}/info.json")
        if info:
            data.append([kind, info["feature_dim"], f"{info['extract_train_seconds']:.3f}", f"{info['extract_val_seconds']:.3f}",
                         f"{info['fit_seconds']:.3f}", f"{info['val_predict_seconds']:.3f}", info["model_file_bytes"], info["stored_examples_bytes"] + info["stored_labels_bytes"]])
    text.append(markdown_table(["Признаки", "Размерность", "Извлечение train, с", "Извлечение val, с", "Fit, с", "Predict val, с", "Файл, байт", "Примеры+метки в RAM, байт"], data))
    text.append("Память kNN определяется прежде всего сохранёнными обучающими векторами и метками; число настраиваемых коэффициентов не описывает её размер. "
                "В info.json также записана память массивов scaler. Таблица не претендует на измерение полного RSS процесса.\n")
    text += ["## 4. Аугментации и абляция\n",
             "Два обучения с нуля: none и rotate10. Начальные веса, seed, train/val и SGD совпадают. "
             "Повороты имеют независимые углы Uniform(−10°, +10°) для каждого train-изображения; bilinear, заполнение 0. "
             "Перемешивание DataLoader и аугментации используют отдельные генераторы с seed=1005. Val/test детерминированы. Flip и другие аугментации отсутствуют. "
             "В обоих запусках ровно 15 эпох; early stopping отключён. Основной вариант — rotate10 независимо от результата сравнения.\n"]
    ablation = rows("figures/ablation.csv")
    text.append(markdown_table(["Политика", "Лучшая эпоха", "Best val accuracy, %", "Last val accuracy, %", "Сумма времени эпох, с"],
                [[r["augmentation"], r["best_epoch"], pct(r["best_val_accuracy"]), pct(r["last_val_accuracy"]), f"{float(r['epoch_seconds_sum']):.3f}"] for r in ablation]))
    delta = js("figures/ablation_delta.json")
    text.append(f"rotate10 − none: best val **{delta['rotate10_minus_none_best_pp']:+.3f} п.п.**, last val **{delta['rotate10_minus_none_last_pp']:+.3f} п.п.**. "
                "Это результат одной фиксированной пары запусков; разброс по seed не измерялся.\n" if delta else "Эффект поворотов не измерен; утверждать улучшение пока нельзя.\n")
    text += ["## 5. Модель, параметры и память\n",
             "Учебная LeNet-5: Conv(1→6, 5×5) → ReLU → MaxPool(2) → Conv(6→16, 5×5) → ReLU → MaxPool(2) → Flatten → "
             "Linear(256→120) → ReLU → Linear(120→84) → ReLU → Linear(84→10). "
             "Это адаптация лекционной схемы к входу 28×28 без padding. После pooling остаётся 16×4×4; AdaptiveAvgPool не требуется. "
             "Свёртки учитывают локальное соседство штрихов и разделяют веса, небольшие слои подходят для CPU. Выход — логиты. "
             "CrossEntropyLoss применяется к логитам; softmax используется только для вероятностей при оценке.\n"]
    architecture = rows("model/architecture.csv")
    text.append(markdown_table(["Слой", "Вход", "Выход", "Параметры"], [[r["layer"], r["input"], r["output"], r["parameters"]] for r in architecture]))
    if model_info:
        text.append(f"Программно получено: **{model_info['parameters']}** параметров, обучаемых **{model_info['trainable_parameters']}**; "
                    f"память параметров float32 — **{model_info['parameter_bytes']} байт** ({model_info['parameter_bytes']/1024:.3f} KiB). "
                    f"Файл начальных необученных весов — {model_info['initial_weights_file_bytes']} байт. Это проверка архитектуры, не результат обучения.\n")
    text.append("[Архитектура](results/model/architecture.txt), [формы](results/model/architecture.csv), [подсчёт](results/model/model_info.json). "
                "best_weights.pt хранит только веса; best.pt и last.pt дополнительно хранят optimizer, scheduler, RNG и историю и поэтому больше.\n")
    text += ["## 6. Обучение, кривые и resume\n",
             "Настройки основных запусков: SGD, lr=0.1, momentum=0.9, weight_decay=5e−4, batch_size=128, "
             "CosineAnnealingLR(T_max=15, eta_min=0), 15 эпох. LR в CSV относится к текущей эпохе; scheduler.step() выполняется после неё. "
             "num_workers=0; CPU float32. Время эпохи включает train и val; throughput = 54 000 / время train-цикла. "
             "Загрузка данных, построение рисунков и запись checkpoints во время эпохи не входят.\n",
             figure("figures/ablation.png", "Loss, accuracy и LR по сохранённым CSV")]
    timings = []
    for name in ["none", "rotate10", "kmnist"]:
        summary = js(f"{name}/summary.json")
        log = rows(f"{name}/log.csv")
        if summary and log:
            timings.append([name, summary["epochs"], f"{summary['train_seconds_sum']:.3f}", f"{summary['epoch_seconds_sum']:.3f}",
                            f"{sum(int(r['train_images']) for r in log)/summary['train_seconds_sum']:.2f}", summary["best_weights_bytes"], summary["best_checkpoint_bytes"]])
    text.append(markdown_table(["Прогон", "Эпох", "Train, с", "Train+val, с", "Train images/s", "Веса, байт", "Best checkpoint, байт"], timings))
    text.append("Запланирован реальный resume: rotate10 завершается после сохранения эпохи 7 и запускается отдельной командой --resume до эпохи 15. "
                "Восстанавливаются веса, momentum SGD, cosine scheduler, номер эпохи, best accuracy, random, NumPy, torch и все четыре генератора. "
                "CSV восстанавливается из истории checkpoint, поэтому строки не дублируются.\n")
    ev = ROOT / "rotate10" / "events.jsonl"
    resumes = [json.loads(line) for line in ev.read_text(encoding="utf-8").splitlines() if json.loads(line)["event"] == "resume"] if ev.exists() else []
    if resumes:
        for r in resumes:
            text.append(f"Зафиксировано продолжение после эпохи {r['restored_epoch']}, следующая эпоха {r['next_epoch']}, "
                        f"LR={r['next_lr']:.9f}, scheduler.last_epoch={r['scheduler_last_epoch']}. [Журнал](results/rotate10/events.jsonl).\n")
    else:
        text.append("**Реальный resume основного MNIST-прогона не выполнен.** Проверка сериализации на синтетических входах его не заменяет.\n")
    text += ["## 7. Метрики, ошибки и robustness\n",
             metrics_table([("raw+kNN / MNIST val", "classical/raw/val/metrics.json"), ("HOG+kNN / MNIST val", "classical/hog/val/metrics.json"),
                            ("none / MNIST val", "evaluation/none/mnist/metrics.json"), ("rotate10 / MNIST val", main + "/metrics.json")]),
             "Top-3 kNN вычисляется по predict_proba. При равных вероятностях выбран детерминированный порядок по возрастанию индекса класса; "
             "такие совпадения для kNN часты. Macro-F1 усредняется по всем десяти классам.\n"]
    for label, path in [("raw+kNN", "classical/raw/val"), ("HOG+kNN", "classical/hog/val"),
                        ("LeNet none", "evaluation/none/mnist"), ("LeNet rotate10", main)]:
        per_class = rows(path + "/per_class.csv")
        if per_class:
            text.append(f"**{label}: per-class на MNIST val.**\n")
            text.append(markdown_table(["Класс", "Precision", "Recall", "F1", "Support"],
                        [[r["class"], *[f"{float(r[k]):.5f}" for k in ["precision", "recall", "f1-score"]], r["support"]] for r in per_class]))
            text.append(figure(path + "/confusion.png", label + ": строки истинные, столбцы предсказанные"))
    confusions = rows(main + "/errors/top_confusions.csv")
    text.append(markdown_table(["Истинный → предсказанный", "Ошибок", "Доля класса, %"],
                [[f"{r['true']} → {r['predicted']}", r["count"], pct(r["fraction_of_true_class"])] for r in confusions]))
    errors = rows(main + "/errors/errors.csv")
    if js(main + "/metrics.json"):
        text.append(f"Всего ошибок основной модели на MNIST val: **{len(errors)}**. "
                    f"[Полный CSV](results/{main}/errors/errors.csv) содержит исходный индекс официального train, истинный класс, предсказание и уверенность. "
                    "Галереи включают все ошибки, по 30 на лист.\n")
        if len(errors) < 10:
            text.append(f"Ошибок меньше 10: доступны только {len(errors)} реальных случаев.\n")
        for page in js(main + "/errors/gallery_index.json") or []:
            text.append(figure(main + "/errors/" + page["file"], "Все ошибки: " + page["file"]))
    else:
        text.append("Полный список ошибок и топ-5 пар не получены. Причины ошибок по отсутствующим изображениям не предполагаются.\n")
    review = rows("analysis/error_review.csv")
    detailed = [r for r in review if r["detailed"].lower() in {"1", "yes", "true"}]
    text.append(markdown_table(["Индекс", "Истинный → предсказанный", "Уверенность", "Наблюдение", "Гипотеза причины"],
                [[r["index"], f"{r['true']} → {r['predicted']}", f"{float(r['confidence']):.4f}", r["observation"], r["hypothesis"]] for r in detailed]))
    groups = sorted({r["group"] for r in review if r["group"]})
    for group in groups:
        chosen = [r for r in review if r["group"] == group]
        text.append(f"Группа «{group}»: {len(chosen)} ошибок; индексы {', '.join(r['index'] for r in chosen)}. "
                    "Индивидуальные наблюдения и гипотезы приведены в [error_review.csv](results/analysis/error_review.csv).\n")
    text.append("Причины формулируются как визуальные гипотезы. Неоднозначный почерк сам по себе не доказывает ошибку разметки. "
                "Уверенность softmax не является гарантией корректности.\n")
    rob = rows(main + "/robustness.csv")
    text.append(markdown_table(["Искажение", "Accuracy, %", "Падение clean−corrupted, п.п."],
                [[r["corruption"], pct(r["accuracy"]), f"{float(r['drop_pp']):+.3f}"] for r in rob]))
    text.append("Один и тот же MNIST val, без дообучения: Gaussian noise σ=0.05 с отдельным seed=1005; Gaussian blur 5×5, "
                "σx=σy=1.0, reflect padding; яркость ×0.7 и ×1.3. Каждый эффект применяется отдельно к [0,1] перед нормализацией, "
                "с ограничением диапазона. Знак минус у падения означал бы улучшение.\n")
    if rob:
        worst = max(rob[1:], key=lambda r: float(r["drop_pp"]))
        text.append(f"Наибольшее наблюдаемое падение — {worst['corruption']}: {float(worst['drop_pp']):.3f} п.п.\n")
    text += ["**Финальная оценка test.** Модели выбираются заранее по val; после фиксации selection.json новые обучения запрещены. "
             "KMNIST endpoint — последняя, десятая эпоха. При повторном запуске законченные test-оценки читаются с диска и не повторяются.\n",
             metrics_table([(f"{domain.upper()} test / {name}", f"final_test/{domain}/{name}/metrics.json")
                            for domain, names in [("mnist", ["raw", "hog", "none", "rotate10", "kmnist"]), ("kmnist", ["rotate10", "kmnist"])] for name in names]),
             "Для каждой полученной test-строки в её папке сохраняются per_class.csv и confusion.png с той же нормировкой. "
             "Test не используется для объяснения val-ошибок или пересмотра решений.\n",
             "## 8. Grad-CAM\n",
             "Целевой слой — conv2 с выходом 16×8×8. В model.eval() вычисляется градиент логита предсказанного класса по его картам. "
             "Градиенты усредняются по пространственным координатам; затем взвешенная сумма карт → ReLU → bilinear resize 28×28 → "
             "нормализация [0,1] → наложение с alpha=0.45. Нулевые карты остаются нулевыми. Градиенты включены. "
             "Из val выбираются первые по исходному индексу пять верных и пять ошибочных предсказаний. "
             "Искусственные ошибки не создаются. Подписаны true, predicted и target; исходные карты сохраняются как .npy.\n",
             figure(main + "/gradcam/correct.png", "Grad-CAM: верные предсказания"),
             figure(main + "/gradcam/wrong.png", "Grad-CAM: ошибочные предсказания")]
    cams = rows("analysis/gradcam_review.csv")
    text.append(markdown_table(["Индекс", "Группа", "True / pred / target", "Наблюдаемые участки", "Гипотеза связи с ответом"],
                [[r["index"], r["group"], f"{r['true']} / {r['predicted']} / {r['target']}", r["observation"], r["hypothesis"]] for r in cams if r["observation"]]))
    text.append("Без фактических карт нельзя утверждать, на какие штрихи реагирует модель. CAM имеет низкое пространственное разрешение и "
                "показывает чувствительность логита, а не причинное доказательство: пятно на фоне может включать рецептивное поле соседнего штриха.\n")
    text += ["## 9. MNIST → KMNIST\n",
             "Источник — неизменённая копия rotate10/best.pt в kmnist/source_mnist.pt. Архитектура и десять выходов головы сохранены. "
             "Фиксированный режим без перебора: 10 эпох, SGD lr=0.01, momentum=0.9, weight_decay=5e−4, batch=128, cosine T_max=10, "
             "повороты ±10° только на KMNIST train. Эпоха 0 и каждая следующая оцениваются на обоих фиксированных val. "
             "Итог для сравнения — last.pt, а не эпоха, выбранная по сохранению MNIST.\n",
             "KMNIST содержит исторические японские символы хираганы, MNIST — цифры. Одинаковый номер класса не означает одинаковый объект. "
             "Начальная KMNIST accuracy характеризует совпадение числовых выходов с новой разметкой, а не распознавание тех же объектов в другом домене. "
             "Снижение MNIST accuracy включает и изменение признаков, и перепрофилирование общей головы. "
             "[Источник KMNIST](https://github.com/rois-codh/kmnist).\n"]
    comparison = rows("tables/mnist_val.csv")
    text.append("**Общая таблица на одном MNIST val.**\n")
    text.append(markdown_table(["Подход", "Accuracy, %", "Macro-F1", "Время обучения, с", "Модель, байт", "Признаки / параметры", "Train-векторы RAM, байт"],
                [[r["model"], pct(r["accuracy"]), f"{float(r['macro_f1']):.5f}", f"{float(r['training_seconds']):.3f}", r["model_bytes"], r["features_or_parameters"], r["stored_examples_bytes"]] for r in comparison]))
    text.append("Для классики время = извлечение train-признаков + fit; для сетей = сумма train+val по эпохам. "
                "Это разные явно указанные границы, не чистое сравнение времени optimizer. Время после KMNIST — дополнительное дообучение; "
                "затраты исходного MNIST-прогона к нему добавляются при оценке полного маршрута. Последняя строка проверяет сохранение исходной задачи.\n")
    km = rows("tables/kmnist_val.csv")
    text.append("**Отдельная таблица на одном KMNIST val.**\n")
    text.append(markdown_table(["Состояние", "Accuracy, %", "Top-3, %", "Macro-F1"],
                [[r["model"], pct(r["accuracy"]), pct(r["top3_accuracy"]), f"{float(r['macro_f1']):.5f}"] for r in km]))
    text.append(figure("kmnist/adaptation.png", "Адаптация и изменение качества исходной задачи по сохранённому domain_log.csv"))
    domain = rows("kmnist/domain_log.csv")
    if len(domain) > 1:
        start, end = domain[0], domain[-1]
        text.append(f"MNIST val: {pct(start['mnist_accuracy'])}% → {pct(end['mnist_accuracy'])}%, изменение "
                    f"{100*(float(end['mnist_accuracy'])-float(start['mnist_accuracy'])):+.3f} п.п. "
                    f"KMNIST val: {pct(start['kmnist_accuracy'])}% → {pct(end['kmnist_accuracy'])}%, изменение "
                    f"{100*(float(end['kmnist_accuracy'])-float(start['kmnist_accuracy'])):+.3f} п.п.\n")
    text.append("Обучения KMNIST с нуля в этом эксперименте нет; вывод о преимуществе переноса над таким обучением делать нельзя.\n")
    text += ["## 10. Latency и экономика\n"]
    lat = js(main + "/latency.json")
    if lat:
        text.append(f"Измерено на {lat['cpu']}: **{lat['mean_ms']:.6f} мс/изображение**, **{lat['fps']:.2f} FPS**. "
                    "[Все 100 измерений](results/evaluation/rotate10/mnist/latency_samples.csv).\n")
    else:
        text.append("Latency обученной LeNet не измерена. Время необученной сети не подставляется вместо требуемого измерения.\n")
    text.append("Протокол: CPU, float32, batch=1, eval(), inference_mode(), 20 прогревочных проходов и 100 разных val-изображений. "
                "Измеряется только model forward на заранее подготовленном нормализованном тензоре; загрузка, нормализация, softmax и запись исключены. "
                "FPS = 1000 / среднее время в мс. Измерение зависит от загрузки и режима питания CPU.\n")
    eco = js("tables/economics.json") or {"power_watts": None, "tariff_per_kwh": None, "items": []}
    text.append(f"Мощность, Вт: {eco['power_watts'] if eco['power_watts'] is not None else '__________'}. "
                f"Тариф за кВт·ч: {eco['tariff_per_kwh'] if eco['tariff_per_kwh'] is not None else '__________'}. "
                "При отсутствии этих исходных данных денежные суммы оставлены пустыми. TDP процессора не подставляется как измеренная мощность.\n")
    text.append("`Стоимость = мощность_Вт / 1000 × (время_с / 3600) × тариф_за_кВт·ч`. "
                "Для 1000 инференсов время_с = 1000 × latency_мс / 1000. Переобучение с тем же бюджетом оценивается по фактическому времени основного обучения; "
                "это расчётная оценка повторного запуска, а не ещё одно выполненное обучение.\n")
    text.append(markdown_table(["Операция", "Время, с", "Время, ч", "Стоимость"],
                [[r["operation"], f"{r['seconds']:.6f}", f"{r['seconds']/3600:.9f}", "" if r["cost"] is None else f"{r['cost']:.9f}"] for r in eco["items"]]))
    text.append("При увеличении данных в 10 раз число шагов при прежних batch и эпохах вырастет примерно в 10 раз; "
                "для kNN также растут память train-векторов и работа поиска соседей на запрос. Дополнительные данные могут улучшить обобщение, "
                "но величина прироста неизвестна. В 10 раз больше параметров означает примерно в 10 раз больше памяти весов в той же precision; "
                "latency зависит от того, какие слои расширены, поэтому пропорциональный рост не гарантирован. Большая сеть может переобучаться.\n")
    text += ["## 11. Выводы\n"]
    metric = js(main + "/metrics.json")
    if metric:
        text.append(f"Основная LeNet достигла {pct(metric['accuracy'])}% на MNIST val, macro-F1={metric['macro_f1']:.5f}; "
                    f"осталось {metric['errors']} ошибок из {metric['n']}. Это достигнутое качество при данном бюджете, а не доказанный абсолютный потолок архитектуры.\n")
    else:
        text.append("Фактическое качество, оставшиеся ошибки, различие моделей по accuracy, эффект аугментации и забывание после KMNIST "
                    "не установлены: датасеты недоступны и полные обучения не выполнены. Утверждать достижение 99% нельзя.\n")
    text.append("В классике raw вручную задаёт пиксельное представление уменьшенного изображения, HOG — локальные гистограммы направлений градиента. "
                "kNN хранит примеры и сравнивает расстояния после масштабирования признаков. LeNet обучает локальные фильтры и сочетания признаков по loss; "
                "ей не задаются готовые направления штрихов. Для определения ограничений качества нужны реальные ошибки, разрыв train/val и карты Grad-CAM.\n")
    if len(comparison) == 4:
        text.append("Численное сравнение на общем MNIST val: " + "; ".join(f"{r['model']} — {pct(r['accuracy'])}%" for r in comparison) + ".\n")
    if delta:
        text.append(f"Измеренный вклад rotate10 относительно none по best val: {delta['rotate10_minus_none_best_pp']:+.3f} п.п.\n")
    text.append("**Проверка выполнения.** [Критерий → конкретные файлы](results/checks/criteria.csv), "
                "[итог проверки](results/checks/final_check.json). "
                "[Синтетические функциональные проверки](results/checks/self_check.json) проверяют код и не заменяют датасетные эксперименты. "
                "Все команды продолжения приведены в [README.md](README.md).\n")
    check = js("checks/final_check.json")
    if check and check["pending"]:
        text.append("**Невыполненные критерии:**\n\n" + "\n".join("- " + name for name in check["pending"]) + "\n")
    Path("report.md").write_text("\n".join(text), encoding="utf-8")
