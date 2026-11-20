import argparse
import json
import pickle
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import cvkit as cv


def prepare():
    cv.save_json(cv.ROOT / "environment.json", cv.environment())
    folder = cv.ROOT / "model"
    folder.mkdir(parents=True, exist_ok=True)
    model = cv.LeNet5()
    info = cv.describe_model(model, folder)
    torch.save(model.state_dict(), folder / "initial_weights.pt")
    info["initial_weights_file_bytes"] = (folder / "initial_weights.pt").stat().st_size
    info["trained"] = False
    cv.save_json(folder / "model_info.json", info)
    cv.save_json(cv.ROOT / "protocol.json", {
        "seed": 1005, "device": "cpu", "threads": torch.get_num_threads(), "precision": "float32",
        "datasets": ["torchvision.datasets.MNIST", "torchvision.datasets.KMNIST"],
        "split": "official train: stratified 90/10, random_state=1005; official test preserved",
        "normalization": "mean and population std from MNIST train only; same for both domains",
        "main": {"aug": "rotate10", "epochs": 15, "batch_size": 128, "lr": 0.1,
                 "optimizer": "SGD", "momentum": 0.9, "weight_decay": 0.0005,
                 "scheduler": "cosine T_max=15", "early_stopping": False},
        "control": "same initialization and settings; aug=none",
        "resume": "rotate10 stops after epoch 7, resumes to epoch 15 in a separate process",
        "finetune": {"epochs": 10, "lr": 0.01, "optimizer": "SGD", "momentum": 0.9,
                     "weight_decay": 0.0005, "batch_size": 128, "scheduler": "cosine T_max=10",
                     "aug": "rotate10", "early_stopping": False, "endpoint": "last.pt"},
        "test_selection": ["raw kNN", "HOG kNN", "none/best.pt", "rotate10/best.pt", "kmnist/last.pt"],
        "power_watts": None, "tariff_per_kwh": None})
    availability = []
    for name in ["mnist", "kmnist"]:
        try:
            info = cv.eda(name)
            if name == "mnist":
                cv.processing_sheet()
            availability.append({"dataset": name, "available": True, "splits": info})
        except (RuntimeError, OSError) as error:
            availability.append({"dataset": name, "available": False, "exception": type(error).__name__, "message": str(error)})
            print(f"{name}: {error}", flush=True)
    cv.save_json(cv.ROOT / "data_availability.json", availability)
    return all(row["available"] for row in availability)


def evaluate_checkpoint(path, dataset, folder, args):
    base, _, val = cv.split_data(dataset)
    model, checkpoint = cv.load_model(path)
    norm = checkpoint["config"]["normalization"]
    result = cv.predict(model, cv.loader(base, val), norm)
    metrics = cv.save_metrics(folder, result["targets"], result["probabilities"], result["indices"])
    cv.save_json(folder / "provenance.json", {"checkpoint": str(path), "checkpoint_sha256": cv.file_hash(path),
                 "epoch": checkpoint["epoch"], "dataset": dataset, "split": "val",
                 "split_sha256": cv.file_hash(cv.ROOT / "data" / f"{dataset}_split.npz"), "normalization": norm})
    if args.errors:
        cv.all_errors(base, result, folder / "errors")
    if args.gradcam:
        cv.gradcam_examples(model, base, result, norm, folder / "gradcam")
    if args.robustness:
        cv.robustness(model, cv.loader(base, val), norm, result["accuracy"], folder)
    if args.latency:
        cv.latency(model, base, val, norm, folder)
    print(dataset, path, metrics, flush=True)
    return metrics


def review_templates():
    folder = cv.ROOT / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    source = cv.ROOT / "evaluation" / "rotate10" / "mnist"
    if (source / "errors" / "errors.csv").exists() and not (folder / "error_review.csv").exists():
        rows = [{**r, "group": "", "observation": "", "hypothesis": "", "detailed": ""}
                for r in cv.read_csv(source / "errors" / "errors.csv")]
        cv.write_csv(folder / "error_review.csv", rows,
                     ["index", "true", "predicted", "confidence", "group", "observation", "hypothesis", "detailed"])
    if (source / "gradcam" / "examples.csv").exists() and not (folder / "gradcam_review.csv").exists():
        rows = [{**r, "observation": "", "hypothesis": ""} for r in cv.read_csv(source / "gradcam" / "examples.csv")]
        cv.write_csv(folder / "gradcam_review.csv", rows)


def review_complete():
    source = cv.ROOT / "evaluation" / "rotate10" / "mnist"
    try:
        errors = cv.read_csv(source / "errors" / "errors.csv")
        review = cv.read_csv(cv.ROOT / "analysis" / "error_review.csv")
        cams = cv.read_csv(source / "gradcam" / "examples.csv")
        cam_review = cv.read_csv(cv.ROOT / "analysis" / "gradcam_review.csv")
        assert {r["index"] for r in errors} == {r["index"] for r in review}
        assert len(errors) == len(review)
        source_rows = {r["index"]: r for r in errors}
        assert all(all(r[key] == source_rows[r["index"]][key] for key in ["true", "predicted", "confidence"]) for r in review)
        assert all(r["group"].strip() and r["observation"].strip() and r["hypothesis"].strip() for r in review)
        assert sum(r["detailed"].lower() in {"1", "yes", "true"} for r in review) >= min(10, len(errors))
        assert {r["index"] for r in cams} == {r["index"] for r in cam_review}
        assert len(cams) == len(cam_review)
        source_cams = {r["index"]: r for r in cams}
        assert all(all(r[key] == source_cams[r["index"]][key] for key in ["true", "predicted", "target", "group"]) for r in cam_review)
        assert all(r["observation"].strip() and r["hypothesis"].strip() for r in cam_review)
        return True
    except (FileNotFoundError, AssertionError, KeyError):
        return False


def final_test():
    folder = cv.ROOT / "final_test"
    selection_path = folder / "selection.json"
    if (folder / "completed.json").exists():
        print("Final test is already complete. Saved results are kept unchanged.")
        return
    models = {"none": cv.ROOT / "none" / "best.pt", "rotate10": cv.ROOT / "rotate10" / "best.pt",
              "kmnist": cv.ROOT / "kmnist" / "last.pt"}
    for name, epochs in [("none", 15), ("rotate10", 15), ("kmnist", 10)]:
        summary = cv.read_json(cv.ROOT / name / "summary.json")
        assert summary["epochs"] == epochs
    required = [cv.ROOT / "classical" / kind / "val" / "metrics.json" for kind in ["raw", "hog"]]
    required += [cv.ROOT / "evaluation" / name / domain / "metrics.json"
                 for name, domain in [("none", "mnist"), ("rotate10", "mnist"), ("kmnist", "mnist"), ("kmnist", "kmnist")]]
    required += [cv.ROOT / "evaluation" / "rotate10" / "mnist" / f for f in ["robustness.csv", "latency.json"]]
    assert all(p.exists() for p in required), "Complete all validation experiments first"
    assert review_complete(), "Complete visual error and Grad-CAM review before final test"
    selected = {"checkpoints": {name: {"path": str(path), "sha256": cv.file_hash(path)} for name, path in models.items()},
                "classical": {k: cv.file_hash(cv.ROOT / "classical" / k / "model.pkl") for k in ["raw", "hog"]},
                "selection_rule": "MNIST best val; KMNIST last epoch 10; choices frozen before test",
                "seed": cv.SEED}
    if selection_path.exists():
        assert cv.read_json(selection_path) == selected
    else:
        cv.save_json(selection_path, selected)
        cv.event(folder / "events.jsonl", "selection_frozen", selection=selected)
    for domain in ["mnist", "kmnist"]:
        base = cv.load_base(domain, False)
        ids = np.arange(len(base))
        if domain == "mnist":
            for kind in ["raw", "hog"]:
                dest = folder / domain / kind
                if (dest / "metrics.json").exists():
                    continue
                with (cv.ROOT / "classical" / kind / "model.pkl").open("rb") as f:
                    model = pickle.load(f)
                start = time.perf_counter()
                x = cv.features(base.data.numpy(), kind)
                extraction_seconds = time.perf_counter() - start
                probs = model.predict_proba(x)
                cv.save_metrics(dest, base.targets.numpy(), probs, ids)
                cv.save_json(dest / "timing.json", {"extract_seconds": extraction_seconds})
                cv.event(folder / "events.jsonl", "evaluated", domain=domain, model=kind)
        names = ["none", "rotate10", "kmnist"] if domain == "mnist" else ["rotate10", "kmnist"]
        for name in names:
            dest = folder / domain / name
            if (dest / "metrics.json").exists():
                continue
            model, checkpoint = cv.load_model(models[name])
            result = cv.predict(model, cv.loader(base, ids), checkpoint["config"]["normalization"])
            cv.save_metrics(dest, result["targets"], result["probabilities"], result["indices"])
            cv.event(folder / "events.jsonl", "evaluated", domain=domain, model=name)
        cv.eda(domain, include_test=True)
    cv.save_json(folder / "completed.json", {"complete": True, "selection_sha256": cv.file_hash(selection_path)})


def self_check():
    folder = cv.ROOT / "checks"
    folder.mkdir(parents=True, exist_ok=True)
    checks = []
    model = cv.LeNet5()
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert sum(p.numel() for p in model.parameters()) == 44426
    checks.append("LeNet logits shape 2x10; 44426 parameters")
    images = np.zeros((10, 28, 28), dtype=np.uint8)
    for i in range(10):
        images[i, 4:24, 4 + i:7 + i] = 255
    raw, hog = cv.features(images, "raw"), cv.features(images, "hog")
    assert raw.shape == (10, 256) and hog.shape == (10, 144)
    assert raw.min() >= 0 and raw.max() <= 1
    checks.append("Raw 256 and HOG 144 features on synthetic shapes")
    pipeline = cv.make_pipeline(cv.StandardScaler(), cv.KNeighborsClassifier(n_neighbors=5, metric="euclidean", algorithm="brute"))
    pipeline.fit(raw, np.arange(10))
    with (folder / "synthetic_knn.pkl").open("wb") as f:
        pickle.dump(pipeline, f)
    with (folder / "synthetic_knn.pkl").open("rb") as f:
        restored = pickle.load(f)
    assert np.array_equal(pipeline.predict_proba(raw), restored.predict_proba(raw))
    assert np.allclose(restored.predict_proba(raw).sum(1), 1)
    checks.append("kNN probabilities and serialization")
    generators = {name: torch.Generator().manual_seed(cv.SEED) for name in ["train", "val", "augmentation", "mnist_val"]}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    x = torch.from_numpy(images).float().unsqueeze(1) / 255
    y = torch.arange(10)

    def step(net, opt, schedule):
        net.train()
        order = torch.randperm(len(x), generator=generators["train"])
        batch = cv.rotate_batch(x[order], generators["augmentation"])
        assert batch.min() >= 0 and batch.max() <= 1
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(net(batch), y[order]).backward()
        opt.step()
        schedule.step()
        return [random.random(), float(np.random.random()), float(torch.rand(()))]

    step(model, optimizer, scheduler)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": cv.rng_state(generators), "epoch": 1}, folder / "synthetic_resume.pt")
    expected_random = step(model, optimizer, scheduler)
    expected_hash, expected_lr = cv.state_hash(model), optimizer.param_groups[0]["lr"]
    saved = torch.load(folder / "synthetic_resume.pt", weights_only=False)
    other = cv.LeNet5()
    other.load_state_dict(saved["model"])
    opt = torch.optim.SGD(other.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    opt.load_state_dict(saved["optimizer"])
    sch.load_state_dict(saved["scheduler"])
    cv.restore_rng(saved["rng"], generators)
    actual_random = step(other, opt, sch)
    assert cv.state_hash(other) == expected_hash and opt.param_groups[0]["lr"] == expected_lr
    assert expected_random == actual_random
    checks.append("Synthetic optimizer/scheduler/RNG resume gives bitwise identical next update")
    cam = cv.gradcam(other, x[:1], int(other(x[:1]).argmax()))
    assert cam.shape == (28, 28) and np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
    checks.append("Grad-CAM autograd works in eval mode")
    for sigma in [0.05]:
        noisy = (x + sigma * torch.randn(x.shape, generator=generators["val"])).clamp(0, 1)
        blurred = cv.gaussian_blur(x, [5, 5], [1.0, 1.0])
        assert noisy.shape == blurred.shape == x.shape
    checks.append("Noise and deterministic 5x5 Gaussian blur APIs")
    cv.save_json(folder / "self_check.json", {"passed": True, "checks": checks,
                 "scope": "synthetic functional checks only; not MNIST training, real resume, accuracy or latency"})
    print(json.dumps(checks, ensure_ascii=False), flush=True)


def summarize(power, tariff):
    if (power is None) != (tariff is None):
        raise ValueError("Supply both --power-watts and --tariff-per-kwh, or neither")
    if power is not None and (power <= 0 or tariff < 0):
        raise ValueError("Power must be positive and tariff nonnegative")
    folder = cv.ROOT / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, label in [("raw", "raw+kNN"), ("hog", "HOG+kNN")]:
        path = cv.ROOT / "classical" / kind
        if (path / "info.json").exists():
            info, metric = cv.read_json(path / "info.json"), cv.read_json(path / "val" / "metrics.json")
            rows.append({"model": label, "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
                         "training_seconds": info["train_total_seconds"], "model_bytes": info["model_file_bytes"],
                         "features_or_parameters": info["feature_dim"], "stored_examples_bytes": info["stored_examples_bytes"]})
    for name, label in [("rotate10", "LeNet scratch rotate10"), ("kmnist", "LeNet after KMNIST: MNIST retention")]:
        path = cv.ROOT / "evaluation" / name / "mnist" / "metrics.json"
        if path.exists():
            metric, summary = cv.read_json(path), cv.read_json(cv.ROOT / name / "summary.json")
            info = cv.read_json(cv.ROOT / name / "model_info.json")
            rows.append({"model": label, "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
                         "training_seconds": summary["epoch_seconds_sum"],
                         "model_bytes": summary["last_weights_bytes" if name == "kmnist" else "best_weights_bytes"],
                         "features_or_parameters": info["parameters"], "stored_examples_bytes": 0})
    if rows:
        cv.write_csv(folder / "mnist_val.csv", rows)
    rows = []
    for label, path in [("before: MNIST model", cv.ROOT / "kmnist" / "epoch0" / "kmnist" / "metrics.json"),
                        ("after: epoch 10", cv.ROOT / "evaluation" / "kmnist" / "kmnist" / "metrics.json")]:
        if path.exists():
            metric = cv.read_json(path)
            rows.append({"model": label, **{k: metric[k] for k in ["accuracy", "top3_accuracy", "macro_f1"]}})
    if rows:
        cv.write_csv(folder / "kmnist_val.csv", rows)
    economics = {"power_watts": power, "tariff_per_kwh": tariff, "currency": None,
                 "formula": "power_watts / 1000 * seconds / 3600 * tariff_per_kwh", "items": []}
    for name in ["none", "rotate10", "kmnist"]:
        path = cv.ROOT / name / "summary.json"
        if path.exists():
            economics["items"].append({"operation": name, "seconds": cv.read_json(path)["epoch_seconds_sum"]})
    main_time = next((r["seconds"] for r in economics["items"] if r["operation"] == "rotate10"), None)
    if main_time is not None:
        economics["items"].append({"operation": "retraining_same_15_epoch_budget", "seconds": main_time})
    latency_path = cv.ROOT / "evaluation" / "rotate10" / "mnist" / "latency.json"
    if latency_path.exists():
        economics["items"].append({"operation": "1000_inferences", "seconds": cv.read_json(latency_path)["mean_ms"]})
    for row in economics["items"]:
        row["cost"] = None if power is None else power / 1000 * row["seconds"] / 3600 * tariff
    cv.save_json(folder / "economics.json", economics)


def check_results():
    criteria = [
        ("0", "Среда, seed, команды, протокол", ["environment.json", "commands.jsonl", "protocol.json"]),
        ("1", "MNIST: EDA и train/val индексы", ["data/mnist_split.npz", "data/normalization.json", "eda/mnist/summary.json", "eda/mnist/class_counts.csv", "eda/mnist/class_hist.png", "eda/mnist/contact_sheet.png"]),
        ("2a", "Лист обработки на 3 изображениях", ["classical/processing.png", "classical/processing.json"]),
        ("2b", "Raw+kNN, полный train", ["classical/raw/model.pkl", "classical/raw/info.json", "classical/raw/val/metrics.json", "classical/raw/val/per_class.csv", "classical/raw/val/confusion.png"]),
        ("2c", "HOG+kNN, полный train", ["classical/hog/model.pkl", "classical/hog/info.json", "classical/hog/val/metrics.json", "classical/hog/val/per_class.csv", "classical/hog/val/confusion.png"]),
        ("3", "Абляция none/rotate10", ["figures/ablation.png", "figures/ablation.csv", "figures/ablation_delta.json"]),
        ("4", "Архитектура, формы, параметры и память", ["model/architecture.csv", "model/model_info.json", "model/initial_weights.pt"]),
        ("5a", "none: 15 эпох, логи, checkpoints", ["none/config.json", "none/log.csv", "none/best.pt", "none/last.pt", "none/summary.json"]),
        ("5b", "rotate10: 15 эпох, логи, checkpoints", ["rotate10/config.json", "rotate10/log.csv", "rotate10/best.pt", "rotate10/last.pt", "rotate10/summary.json", "rotate10/events.jsonl"]),
        ("5c", "Реальное продолжение основного прогона", ["rotate10/events.jsonl"]),
        ("6a", "Все ошибки, пары, метрики", ["evaluation/rotate10/mnist/metrics.json", "evaluation/rotate10/mnist/per_class.csv", "evaluation/rotate10/mnist/confusion.png", "evaluation/rotate10/mnist/errors/errors.csv", "evaluation/rotate10/mnist/errors/gallery_index.json", "evaluation/rotate10/mnist/errors/top_confusions.csv", "analysis/error_review.csv"]),
        ("6b", "Robustness: четыре отдельных искажения", ["evaluation/rotate10/mnist/robustness.csv", "evaluation/rotate10/mnist/robustness_config.json"]),
        ("7", "Grad-CAM и визуальный разбор", ["evaluation/rotate10/mnist/gradcam/examples.csv", "evaluation/rotate10/mnist/gradcam/correct.png", "analysis/gradcam_review.csv"]),
        ("8", "KMNIST: исходная модель, эпоха 0, адаптация/забывание", ["data/kmnist_split.npz", "eda/kmnist/summary.json", "kmnist/source_mnist.pt", "kmnist/domain_log.csv", "kmnist/config.json", "kmnist/summary.json", "kmnist/epoch0/mnist/metrics.json", "kmnist/epoch0/kmnist/metrics.json", "kmnist/adaptation.png", "tables/mnist_val.csv", "tables/kmnist_val.csv"]),
        ("9a", "Latency CPU: 20+100", ["evaluation/rotate10/mnist/latency.json", "evaluation/rotate10/mnist/latency_samples.csv"]),
        ("9b", "Экономика с явными исходными данными", ["tables/economics.json"]),
        ("9c", "Единый финальный test", ["final_test/selection.json", "final_test/completed.json", "eda/mnist/class_counts.csv", "eda/kmnist/class_counts.csv"]),
    ]
    rows = []
    for stage, criterion, paths in criteria:
        missing = [str(cv.ROOT / p) for p in paths if not (cv.ROOT / p).exists()]
        rows.append({"stage": stage, "criterion": criterion, "status": "present" if not missing else "pending",
                     "files": "; ".join(str(cv.ROOT / p) for p in paths), "details": "; ".join(missing)})
    problems = []
    for name, count in [("none", 15), ("rotate10", 15), ("kmnist", 10)]:
        path = cv.ROOT / name / "log.csv"
        if not path.exists():
            continue
        log = cv.read_csv(path)
        if [int(r["epoch"]) for r in log] != list(range(1, count + 1)):
            problems.append(f"{name}: incomplete or duplicate epochs")
        cfg = cv.read_json(cv.ROOT / name / "config.json")
        if cfg["early_stopping"] or cfg["seed"] != 1005 or cfg["device"] != "cpu":
            problems.append(f"{name}: invalid fixed settings")
        for r in log:
            expected = cfg["lr"] * (1 + np.cos(np.pi * (int(r["epoch"]) - 1) / count)) / 2
            if not np.isclose(float(r["lr"]), expected, atol=1e-12):
                problems.append(f"{name}: scheduler restarted or incorrect")
            if int(r["train_images"]) != 54000:
                problems.append(f"{name}: training data shortened")
        checkpoint = torch.load(cv.ROOT / name / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != count or checkpoint["scheduler"]["last_epoch"] != count:
            problems.append(f"{name}: checkpoint epoch mismatch")
        if len(checkpoint["history"]) != len(log):
            problems.append(f"{name}: checkpoint and CSV differ")
    if all((cv.ROOT / n / "model_info.json").exists() for n in ["none", "rotate10"]):
        if cv.read_json(cv.ROOT / "none" / "model_info.json")["initial_weights_sha256"] != cv.read_json(cv.ROOT / "rotate10" / "model_info.json")["initial_weights_sha256"]:
            problems.append("Main experiments have different initial weights")
    events = cv.ROOT / "rotate10" / "events.jsonl"
    if events.exists():
        history = [json.loads(s) for s in events.read_text(encoding="utf-8").splitlines()]
        stopped = [r for r in history if r["event"] == "intentional_stop"]
        resumed = [r for r in history if r["event"] == "resume"]
        valid = any(s["saved_epoch"] == r["restored_epoch"] and r["next_epoch"] == s["saved_epoch"] + 1
                    and r["scheduler_last_epoch"] == r["restored_epoch"] and r["optimizer_buffers"] > 0
                    for s in stopped for r in resumed)
        if not valid:
            row = next(row for row in rows if row["stage"] == "5c")
            row["status"] = "pending"
            row["details"] = "No confirmed real stop/resume pair"
    for kind, dim in [("raw", 256), ("hog", 144)]:
        path = cv.ROOT / "classical" / kind / "info.json"
        if path.exists():
            info = cv.read_json(path)
            if info["train_n"] != 54000 or info["feature_dim"] != dim or info["stored_examples_bytes"] < 54000 * dim * 4:
                problems.append(f"{kind}: dataset size, features or kNN memory invalid")
    main = cv.ROOT / "evaluation" / "rotate10" / "mnist"
    if (main / "errors" / "gallery_index.json").exists():
        errors = cv.read_csv(main / "errors" / "errors.csv")
        gallery = cv.read_json(main / "errors" / "gallery_index.json")
        shown = [i for page in gallery for i in page["indices"]]
        expected = [int(r["index"]) for r in errors]
        if sorted(shown) != sorted(expected) or len(shown) != len(set(shown)):
            problems.append("Error galleries omit or duplicate errors")
        if cv.read_json(main / "metrics.json")["errors"] != len(errors):
            problems.append("Error count differs from metrics")
        for page in gallery:
            if not (main / "errors" / page["file"]).exists():
                problems.append("Missing gallery image")
    if (main / "latency.json").exists():
        info = cv.read_json(main / "latency.json")
        if (info["warmup"], info["measured"], info["batch_size"], info["device"]) != (20, 100, 1, "cpu"):
            problems.append("Latency protocol mismatch")
        samples = cv.read_csv(main / "latency_samples.csv")
        if len(samples) != 100 or not np.isclose(np.mean([float(r["milliseconds"]) for r in samples]), info["mean_ms"]):
            problems.append("Latency summary differs from the 100 samples")
    if (main / "robustness.csv").exists():
        robust = cv.read_csv(main / "robustness.csv")
        if [r["corruption"] for r in robust] != ["clean", "noise", "blur", "brightness07", "brightness13"]:
            problems.append("Robustness variants missing")
        for r in robust:
            if not np.isclose(float(r["drop_pp"]), 100 * (float(robust[0]["accuracy"]) - float(r["accuracy"]))):
                problems.append("Robustness drop is inconsistent")
    if (main / "gradcam" / "examples.csv").exists():
        cams = cv.read_csv(main / "gradcam" / "examples.csv")
        metric = cv.read_json(main / "metrics.json")
        counts = {key: sum(r["group"] == key for r in cams) for key in ["correct", "wrong"]}
        if counts != {"correct": min(5, metric["n"] - metric["errors"]), "wrong": min(5, metric["errors"])}:
            problems.append("Grad-CAM example counts differ from 5 correct and up to 5 real errors")
        for r in cams:
            path = main / "gradcam" / f"cam_{r['index']}.npy"
            if not path.exists():
                problems.append("Grad-CAM array missing")
            else:
                cam = np.load(path)
                if cam.shape != (28, 28) or not np.isfinite(cam).all() or cam.min() < 0 or cam.max() > 1:
                    problems.append("Grad-CAM array invalid")
    for name in ["mnist", "kmnist"]:
        split = cv.ROOT / "data" / f"{name}_split.npz"
        if split.exists():
            with np.load(split) as data:
                if len(data["train"]) != 54000 or len(data["val"]) != 6000 or not np.array_equal(np.sort(np.r_[data["train"], data["val"]]), np.arange(60000)):
                    problems.append(f"{name}: split sizes or partition invalid")
        for path in [cv.ROOT / "evaluation" / n / name for n in ["none", "rotate10", "kmnist"]]:
            if (path / "predictions.npz").exists():
                with np.load(path / "predictions.npz") as data, np.load(split) as expected:
                    if not np.array_equal(data["indices"], expected["val"]):
                        problems.append(f"{path}: validation split differs")
    if (cv.ROOT / "kmnist" / "domain_log.csv").exists():
        domain = cv.read_csv(cv.ROOT / "kmnist" / "domain_log.csv")
        if [int(r["epoch"]) for r in domain] != list(range(11)):
            problems.append("KMNIST monitoring must include epochs 0 through 10")
        if (cv.ROOT / "kmnist" / "source_mnist.pt").exists():
            if cv.file_hash(cv.ROOT / "kmnist" / "source_mnist.pt") != cv.file_hash(cv.ROOT / "rotate10" / "best.pt"):
                problems.append("KMNIST source differs from frozen main MNIST checkpoint")
        for domain_name in ["mnist", "kmnist"]:
            endpoint = cv.ROOT / "evaluation" / "kmnist" / domain_name / "metrics.json"
            if endpoint.exists() and not np.isclose(cv.read_json(endpoint)["accuracy"], float(domain[-1][f"{domain_name}_accuracy"])):
                problems.append("Transfer endpoint differs from last epoch")
    if (cv.ROOT / "tables" / "mnist_val.csv").exists() and len(cv.read_csv(cv.ROOT / "tables" / "mnist_val.csv")) != 4:
        problems.append("Common MNIST val table must contain four models")
    metric_paths = list((cv.ROOT / "evaluation").glob("*/*/metrics.json"))
    metric_paths += list((cv.ROOT / "classical").glob("*/val/metrics.json"))
    metric_paths += list((cv.ROOT / "final_test").glob("*/*/metrics.json"))
    for path in metric_paths:
        data = np.load(path.parent / "predictions.npz")
        y, prob = data["targets"], data["probabilities"]
        metric = cv.read_json(path)
        if len(y) != metric["n"] or prob.shape != (len(y), 10) or not np.allclose(prob.sum(1), 1, atol=1e-6):
            problems.append(f"{path}: probabilities or count invalid")
        if not np.isclose((prob.argmax(1) == y).mean(), metric["accuracy"]):
            problems.append(f"{path}: accuracy does not match predictions")
        for filename in ["per_class.csv", "classification_report.json", "confusion_counts.csv", "confusion_normalized.csv", "confusion.png"]:
            if not (path.parent / filename).exists():
                problems.append(f"{path.parent}: {filename} missing")
        data.close()
    if (cv.ROOT / "final_test" / "completed.json").exists():
        selection = cv.read_json(cv.ROOT / "final_test" / "selection.json")
        for model in selection["checkpoints"].values():
            if cv.file_hash(model["path"]) != model["sha256"]:
                problems.append("Model changed after final test selection")
        for domain, names in [("mnist", ["raw", "hog", "none", "rotate10", "kmnist"]), ("kmnist", ["rotate10", "kmnist"])]:
            for name in names:
                path = cv.ROOT / "final_test" / domain / name / "metrics.json"
                if not path.exists() or cv.read_json(path)["n"] != 10000:
                    problems.append(f"Final {domain}/{name} missing or truncated")
            if sum(int(r["count"]) for r in cv.read_csv(cv.ROOT / "eda" / domain / "class_counts.csv") if r["split"] == "test") != 10000:
                problems.append(f"{domain}: final test class counts missing")
    if not review_complete():
        for row in rows:
            if row["stage"] in {"6a", "7"}:
                row["status"] = "pending"
                row["details"] += "; visual interpretation not complete"
    economics_path = cv.ROOT / "tables" / "economics.json"
    if economics_path.exists():
        economics = cv.read_json(economics_path)
        needed = {"none", "rotate10", "kmnist", "retraining_same_15_epoch_budget", "1000_inferences"}
        if {r["operation"] for r in economics["items"]} != needed:
            row = next(row for row in rows if row["stage"] == "9b")
            row["status"] = "pending"
            row["details"] = "Formula is ready, measured training and inference times are missing"
    complete = not problems and all(r["status"] == "present" for r in rows)
    cv.write_csv(cv.ROOT / "checks" / "criteria.csv", rows)
    cv.save_json(cv.ROOT / "checks" / "final_check.json", {"complete": complete, "problems": problems,
                 "pending": [r["criterion"] for r in rows if r["status"] != "present"],
                 "meaning": "Artifact/consistency checks; visual interpretations require human inspection"})
    print(f"Complete: {complete}; present {sum(r['status']=='present' for r in rows)}/{len(rows)}; problems={problems}")
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--classical", choices=["raw", "hog", "all"])
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--dataset", choices=["mnist", "kmnist"], default="mnist")
    parser.add_argument("--model", choices=["lenet"], default="lenet")
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--seed", choices=[1005], type=int, default=1005)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--processing", action="store_true")
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--review-templates", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--power-watts", type=float)
    parser.add_argument("--tariff-per-kwh", type=float)
    args = parser.parse_args()
    socket.setdefaulttimeout(30)
    cv.setup(args.threads)
    cv.record_command()
    if args.self_check:
        self_check()
    if args.prepare and not prepare():
        sys.exit(2)
    if args.processing:
        cv.processing_sheet()
    if args.classical:
        if (cv.ROOT / "final_test" / "selection.json").exists():
            raise RuntimeError("Models frozen after final test selection")
        for kind in (["raw", "hog"] if args.classical == "all" else [args.classical]):
            cv.classical_fit(kind)
    if args.ckpt:
        if (args.errors or args.gradcam or args.robustness or args.latency) and args.dataset != "mnist":
            parser.error("Detailed diagnostics belong to the main MNIST model")
        folder = args.out_dir or cv.ROOT / "evaluation" / args.ckpt.parent.name / args.dataset
        evaluate_checkpoint(args.ckpt, args.dataset, folder, args)
    if args.plots:
        cv.plot_training()
        if (cv.ROOT / "kmnist" / "domain_log.csv").exists():
            cv.plot_transfer()
    if args.review_templates:
        review_templates()
    if args.final_test:
        final_test()
    if args.report:
        summarize(args.power_watts, args.tariff_per_kwh)
    if args.check:
        complete = check_results()
        if args.report:
            cv.write_report()
        sys.exit(0 if complete else 2)
    if args.report:
        cv.write_report()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        cv.event(cv.ROOT / "failures.jsonl", "failure", command=sys.argv, exception=type(error).__name__, message=str(error))
        raise
