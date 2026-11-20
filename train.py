import argparse
import shutil
import socket
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import cvkit as cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "kmnist"], default="mnist")
    parser.add_argument("--model", choices=["lenet"], default="lenet")
    parser.add_argument("--aug", choices=["none", "rotate10"], default="rotate10")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=["sgd"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["cosine"], default="cosine")
    parser.add_argument("--seed", type=int, choices=[1005], default=1005)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    socket.setdefaulttimeout(30)
    cv.setup(args.threads)
    cv.record_command()
    if (cv.ROOT / "final_test" / "selection.json").exists():
        raise RuntimeError("Final test has started. Experiments are frozen.")
    if args.dataset == "mnist":
        assert (args.epochs, args.batch_size, args.lr, args.momentum, args.weight_decay) == (15, 128, 0.1, 0.9, 5e-4)
        assert args.init is None
    else:
        assert args.aug == "rotate10"
        if args.init is None and args.resume is None:
            parser.error("KMNIST requires --init results/rotate10/best.pt or --resume")
    if args.stop_after is not None and not 1 <= args.stop_after < args.epochs:
        parser.error("--stop-after must be between 1 and epochs-1")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and args.resume.resolve() != (out / "last.pt").resolve():
        parser.error("Resume must use last.pt of the same output directory")
    if args.resume is None and (out / "last.pt").exists():
        raise FileExistsError(f"Use --resume {out / 'last.pt'}")
    base, train, val = cv.split_data(args.dataset)
    norm = cv.normalization()
    generators = {n: torch.Generator().manual_seed(cv.SEED) for n in ["train", "val", "augmentation", "mnist_val"]}
    train_loader = cv.loader(base, train, args.batch_size, True, generators["train"])
    val_loader = cv.loader(base, val, args.batch_size, False, generators["val"])
    model = cv.LeNet5()
    config = {"dataset": args.dataset, "model": "LeNet5", "aug": args.aug,
              "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "SGD",
              "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay,
              "scheduler": "CosineAnnealingLR", "T_max": args.epochs, "eta_min": 0,
              "seed": cv.SEED, "device": "cpu", "precision": "float32", "threads": args.threads,
              "num_workers": 0, "early_stopping": False, "normalization": norm,
              "rotation": {"range_degrees": [-10, 10], "interpolation": "bilinear", "fill": 0,
                           "independent_per_image": True, "align_corners": False},
              "split_sha256": cv.file_hash(cv.ROOT / "data" / f"{args.dataset}_split.npz"),
              "selection": "best val accuracy; ties keep earlier epoch",
              "transfer_endpoint": "last epoch, fixed budget" if args.dataset == "kmnist" else None,
              "throughput_scope": "train images / train loop wall seconds",
              "epoch_time_scope": "train loop plus validation on all monitored domains"}
    if args.resume:
        old = cv.read_json(out / "config.json")
        for key, value in config.items():
            if old[key] != value:
                raise ValueError(f"Resume config mismatch: {key}")
        config = old
    else:
        if args.init:
            source, source_checkpoint = cv.load_model(args.init)
            assert source_checkpoint["config"]["dataset"] == "mnist"
            assert source_checkpoint["config"]["aug"] == "rotate10"
            model.load_state_dict(source.state_dict())
            assert norm == source_checkpoint["config"]["normalization"]
            shutil.copyfile(args.init, out / "source_mnist.pt")
            config["source_sha256"] = cv.file_hash(args.init)
            config["source_epoch"] = source_checkpoint["epoch"]
        cv.save_json(out / "config.json", config)
        cv.save_json(out / "environment.json", cv.environment())
        cv.describe_model(model, out)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, domain_history = [], []
    start_epoch, best_accuracy, best_epoch = 1, -1.0, 0
    mnist_loader = None
    if args.dataset == "kmnist":
        mnist, _, mnist_val = cv.split_data("mnist")
        mnist_loader = cv.loader(mnist, mnist_val, args.batch_size, generator=generators["mnist_val"])
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert checkpoint["config"] == config
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy, best_epoch = checkpoint["best_accuracy"], checkpoint["best_epoch"]
        history, domain_history = checkpoint["history"], checkpoint["domain_history"]
        cv.restore_rng(checkpoint["rng"], generators)
        cv.write_csv(out / "log.csv", history)
        if domain_history:
            cv.write_csv(out / "domain_log.csv", domain_history)
        cv.event(out / "events.jsonl", "resume", checkpoint=str(args.resume), checkpoint_sha256=cv.file_hash(args.resume),
                 restored_epoch=checkpoint["epoch"], next_epoch=start_epoch, next_lr=optimizer.param_groups[0]["lr"],
                 scheduler_last_epoch=scheduler.last_epoch, optimizer_buffers=len(optimizer.state),
                 best_accuracy=best_accuracy, restored_rng=["random", "numpy", "torch", *generators])
        print(f"RESUME epoch={start_epoch}, lr={optimizer.param_groups[0]['lr']:.9f}", flush=True)
    else:
        cv.event(out / "events.jsonl", "start", config=config, initial_weights_sha256=cv.state_hash(model))
        if mnist_loader:
            row = {"epoch": 0}
            for domain, data_loader in [("mnist", mnist_loader), ("kmnist", val_loader)]:
                result = cv.predict(model, data_loader, norm)
                cv.save_metrics(out / "epoch0" / domain, result["targets"], result["probabilities"], result["indices"])
                row[f"{domain}_accuracy"] = result["accuracy"]
                row[f"{domain}_loss"] = result["loss"]
            domain_history.append(row)
            cv.write_csv(out / "domain_log.csv", domain_history)
    end_epoch = args.stop_after or args.epochs
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss, correct, count = 0.0, 0, 0
        lr = optimizer.param_groups[0]["lr"]
        for x, y, _ in train_loader:
            if args.aug == "rotate10":
                x = cv.rotate_batch(x, generators["augmentation"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(cv.normalize(x, norm))
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
        train_seconds = time.perf_counter() - epoch_start
        result = cv.predict(model, val_loader, norm)
        if mnist_loader:
            original = cv.predict(model, mnist_loader, norm)
            domain_history.append({"epoch": epoch, "mnist_accuracy": original["accuracy"], "mnist_loss": original["loss"],
                                   "kmnist_accuracy": result["accuracy"], "kmnist_loss": result["loss"]})
        seconds = time.perf_counter() - epoch_start
        row = {"epoch": epoch, "train_loss": train_loss / count, "train_accuracy": correct / count,
               "val_loss": result["loss"], "val_accuracy": result["accuracy"], "lr": lr,
               "epoch_seconds": seconds, "train_seconds": train_seconds, "throughput": count / train_seconds,
               "train_images": count}
        history.append(row)
        scheduler.step()
        improved = result["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch = result["accuracy"], epoch
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": epoch, "best_accuracy": best_accuracy, "best_epoch": best_epoch, "config": config,
                      "rng": cv.rng_state(generators), "history": history, "domain_history": domain_history}
        if improved:
            torch.save(checkpoint, out / "best.pt")
            torch.save(model.state_dict(), out / "best_weights.pt")
        torch.save(checkpoint, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")
        cv.write_csv(out / "log.csv", history)
        if domain_history:
            cv.write_csv(out / "domain_log.csv", domain_history)
        cv.event(out / "events.jsonl", "epoch_saved", epoch=epoch, next_lr=optimizer.param_groups[0]["lr"],
                 val_accuracy=result["accuracy"], last_sha256=cv.file_hash(out / "last.pt"))
        print(f"{args.dataset}/{args.aug} epoch {epoch:02d}/{args.epochs}: "
              f"train={correct/count:.5f} val={result['accuracy']:.5f} "
              f"loss={result['loss']:.5f} lr={lr:.6f} seconds={seconds:.2f}", flush=True)
    if end_epoch < args.epochs:
        cv.event(out / "events.jsonl", "intentional_stop", saved_epoch=end_epoch, target_epochs=args.epochs)
        print(f"STOP saved epoch {end_epoch}; resume in a separate process", flush=True)
    else:
        torch.save(model.state_dict(), out / "last_weights.pt")
        cv.save_json(out / "summary.json", {"epochs": len(history), "best_epoch": best_epoch, "best_accuracy": best_accuracy,
                     "epoch_seconds_sum": sum(r["epoch_seconds"] for r in history),
                     "train_seconds_sum": sum(r["train_seconds"] for r in history),
                     "best_checkpoint_bytes": (out / "best.pt").stat().st_size,
                     "best_weights_bytes": (out / "best_weights.pt").stat().st_size,
                     "last_weights_bytes": (out / "last_weights.pt").stat().st_size})
        cv.event(out / "events.jsonl", "completed", epochs=args.epochs)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        cv.event(cv.ROOT / "failures.jsonl", "failure", command=sys.argv, exception=type(error).__name__, message=str(error))
        raise
