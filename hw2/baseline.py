import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch import nn

from models import EXPECTED_SHAPES, ResNet56


ROOT = Path(__file__).resolve().parent
TRAIN_SIZE = 50_000
TEST_SIZE = 10_000


def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def lr_factor(iteration):
    factor = 1.0
    if iteration >= 50_000:
        factor *= 0.1
    if iteration >= 70_000:
        factor *= 0.1
    return factor


def triangular_lr(iteration, base_lr, max_lr, stepsize):
    cycle = math.floor(1 + iteration / (2 * stepsize))
    x = abs(iteration / stepsize - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0.0, 1.0 - x)


def linear_lr(iteration, base_lr, max_lr, max_iter):
    if max_iter <= 1:
        return max_lr
    t = min(iteration, max_iter - 1) / (max_iter - 1)
    return base_lr + (max_lr - base_lr) * t


def weight_lr(args, iteration):
    schedule = getattr(args, "schedule", "piecewise")
    if schedule == "piecewise":
        return args.base_lr * lr_factor(iteration)
    if schedule == "triangular":
        return triangular_lr(iteration, args.base_lr, args.max_lr, args.stepsize)
    if schedule == "linear":
        return linear_lr(iteration, args.base_lr, args.max_lr, args.max_iter)
    raise ValueError(schedule)


def apply_lr(optimizer, lr_mults, shown_lr):
    for group, mult in zip(optimizer.param_groups, lr_mults):
        group["lr"] = shown_lr * mult
    return shown_lr


def param_groups(model, base_lr, weight_decay):
    buckets = {
        "conv_weight": [],
        "conv_bias": [],
        "bn": [],
        "fc_weight": [],
        "fc_bias": [],
    }
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            buckets["conv_weight"].append(module.weight)
            buckets["conv_bias"].append(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            buckets["bn"].append(module.weight)
            buckets["bn"].append(module.bias)
        elif isinstance(module, nn.Linear):
            buckets["fc_weight"].append(module.weight)
            buckets["fc_bias"].append(module.bias)
    specs = [
        ("conv_weight", 1.0, 1.0),
        ("conv_bias", 2.0, 0.0),
        ("bn", 1.0, 1.0),
        ("fc_weight", 1.0, 1.0),
        ("fc_bias", 2.0, 1.0),
    ]
    groups = []
    for key, lr_mult, decay_mult in specs:
        params = buckets[key]
        groups.append(
            {
                "params": params,
                "lr": base_lr * lr_mult,
                "weight_decay": weight_decay * decay_mult,
            }
        )
    covered = [p for group in groups for p in group["params"]]
    own = [p for p in model.parameters()]
    if len(covered) != len(own) or {id(p) for p in covered} != {id(p) for p in own}:
        raise RuntimeError("optimizer groups do not cover the model")
    return groups


def load_cifar(data_dir):
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    if len(train_set) != TRAIN_SIZE or len(test_set) != TEST_SIZE:
        raise RuntimeError("unexpected cifar-10 size")

    train_images = torch.from_numpy(train_set.data).permute(0, 3, 1, 2).contiguous().float()
    test_images = torch.from_numpy(test_set.data).permute(0, 3, 1, 2).contiguous().float()
    train_labels = torch.tensor(train_set.targets, dtype=torch.long)
    test_labels = torch.tensor(test_set.targets, dtype=torch.long)
    mean = train_images.mean(dim=0)
    train_images -= mean
    test_images -= mean
    return train_images, train_labels, test_images, test_labels


def open_log(path, fresh):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fresh or not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["iteration", "lr", "train_loss", "test_loss", "test_acc", "elapsed_sec"]
            )
        return set(), -1.0, None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    logged = set()
    best_acc = -1.0
    best_iter = None
    for row in rows:
        iteration = int(row["iteration"])
        acc = float(row["test_acc"])
        logged.add(iteration)
        if acc > best_acc:
            best_acc = acc
            best_iter = iteration
    return logged, best_acc, best_iter


def append_log(path, iteration, lr, train_loss, test_loss, test_acc, elapsed):
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        train_text = "" if train_loss is None else f"{train_loss:.6f}"
        writer.writerow(
            [
                iteration,
                f"{lr:.8f}",
                train_text,
                f"{test_loss:.6f}",
                f"{test_acc:.6f}",
                f"{elapsed:.3f}",
            ]
        )


def save_checkpoint(path, next_iteration, model, optimizer, perm):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.stash_bn()
    torch.save(
        {
            "next_iteration": next_iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "perm": perm,
            "cpu_rng": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.restore_bn()
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["cpu_rng"])
    return int(checkpoint["next_iteration"]), checkpoint["perm"]


@torch.inference_mode()
def evaluate(model, images, labels, batch_size, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for start in range(0, images.size(0), batch_size):
        batch_images = images[start : start + batch_size].to(device)
        batch_labels = labels[start : start + batch_size].to(device)
        logits = model(batch_images)
        loss_sum += criterion(logits, batch_labels).item()
        correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        seen += batch_labels.size(0)
    model.train()
    return loss_sum / seen, correct / seen


def train(args):
    schedule = getattr(args, "schedule", "piecewise")
    run_name = getattr(args, "run_name", None) or {
        "piecewise": "baseline",
        "triangular": "clr",
        "linear": "range",
    }[schedule]
    if TRAIN_SIZE % args.batch_size != 0 or TEST_SIZE % args.batch_size != 0:
        raise SystemExit("batch size must divide 50000 and 10000")
    steps_per_epoch = TRAIN_SIZE // args.batch_size

    device = pick_device(args.device)
    print("device", device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_images, train_labels, test_images, test_labels = load_cifar(args.data_dir)
    model = ResNet56(bn_maf=args.bn_maf)
    model.eval()
    shapes = model.trace(torch.zeros(1, 3, 32, 32))
    model.train()
    for name, shape in shapes.items():
        expected = EXPECTED_SHAPES[name]
        if shape != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {shape}")
    model.to(device)

    groups = param_groups(model, args.base_lr, args.weight_decay)
    optimizer = torch.optim.SGD(groups, lr=args.base_lr, momentum=args.momentum)
    lr_mults = [group["lr"] / args.base_lr for group in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.results_dir / f"{run_name}.csv"
    checkpoint_path = args.results_dir / f"{run_name}.pt"

    if checkpoint_path.exists() and not args.resume and not args.fresh:
        raise SystemExit("checkpoint exists, use --resume or --fresh")

    start_iteration = 0
    perm = None
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"no checkpoint: {checkpoint_path}")
        start_iteration, perm = load_checkpoint(checkpoint_path, model, optimizer, device)
        if start_iteration > args.max_iter:
            raise SystemExit(f"checkpoint iteration {start_iteration} > max_iter {args.max_iter}")
        logged, best_acc, best_iter = open_log(log_path, fresh=False)
    else:
        logged, best_acc, best_iter = open_log(log_path, fresh=True)
        if args.fresh and checkpoint_path.exists():
            checkpoint_path.unlink()

    run_started = time.perf_counter()
    loss_since_test = 0.0
    batches_since_test = 0

    def record_test(iteration, train_loss):
        nonlocal best_acc, best_iter, loss_since_test, batches_since_test
        if iteration in logged:
            loss_since_test = 0.0
            batches_since_test = 0
            return
        shown_iter = min(iteration, max(args.max_iter - 1, 0))
        shown_lr = apply_lr(optimizer, lr_mults, weight_lr(args, shown_iter))
        test_loss, acc = evaluate(
            model, test_images, test_labels, args.batch_size, device
        )
        append_log(
            log_path,
            iteration,
            shown_lr,
            train_loss,
            test_loss,
            acc,
            time.perf_counter() - run_started,
        )
        logged.add(iteration)
        print(f"test {iteration}  loss {test_loss:.4f}  acc {acc * 100:.2f}")
        if acc > best_acc:
            best_acc = acc
            best_iter = iteration
        loss_since_test = 0.0
        batches_since_test = 0

    if start_iteration < args.max_iter:
        iteration = start_iteration
        try:
            for iteration in range(start_iteration, args.max_iter):
                shown_lr = apply_lr(optimizer, lr_mults, weight_lr(args, iteration))
                if iteration > 0 and iteration % args.test_interval == 0:
                    mean_loss = None
                    if batches_since_test:
                        mean_loss = loss_since_test / batches_since_test
                    record_test(iteration, mean_loss)

                if iteration % steps_per_epoch == 0:
                    perm = torch.randperm(TRAIN_SIZE)
                if perm is None:
                    raise RuntimeError("missing train permutation")

                offset = (iteration % steps_per_epoch) * args.batch_size
                index = perm[offset : offset + args.batch_size]
                images = train_images[index]
                flip = torch.rand(images.size(0)) < 0.5
                images = images.clone()
                images[flip] = images[flip].flip(-1)
                images = images.to(device)
                labels = train_labels[index].to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                loss_since_test += loss_value
                batches_since_test += 1
                if (iteration + 1) % args.log_interval == 0:
                    print(
                        f"{iteration + 1}/{args.max_iter}  lr {shown_lr:.4f}  loss {loss_value:.4f}"
                    )
                if (iteration + 1) % args.test_interval == 0:
                    save_checkpoint(checkpoint_path, iteration + 1, model, optimizer, perm)
        except KeyboardInterrupt:
            save_checkpoint(checkpoint_path, iteration, model, optimizer, perm)
            print(getattr(args, "resume_hint", "python hw2/baseline.py --resume"))
            raise

    mean_loss = None
    if batches_since_test:
        mean_loss = loss_since_test / batches_since_test
    record_test(args.max_iter, mean_loss)
    save_checkpoint(checkpoint_path, args.max_iter, model, optimizer, perm)
    print(f"best {best_acc * 100:.2f} at {best_iter}")
    print("csv:", log_path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule", choices=("piecewise", "triangular", "linear"), default="piecewise"
    )
    parser.add_argument("--max-iter", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=125)
    parser.add_argument("--base-lr", type=float, default=0.35)
    parser.add_argument("--max-lr", type=float, default=3.0)
    parser.add_argument("--stepsize", type=int, default=5000)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bn-maf", type=float, default=0.999)
    parser.add_argument("--test-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-hint", default="python hw2/baseline.py --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
