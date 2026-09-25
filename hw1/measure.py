import csv
import threading
import time
from pathlib import Path

import numpy as np
import torch

from models import SmallCNN


def load_done(path, fields):
    done = set()
    if not path.exists() or path.stat().st_size == 0:
        return done
    raw = path.read_text().strip().splitlines()
    if not raw:
        return done
    if raw[0].startswith("S"):
        reader = csv.DictReader(raw)
    else:
        reader = csv.DictReader(raw, fieldnames=fields)
    for row in reader:
        done.add((int(row["S"]), int(row["B"])))
    return done


def measure_one(model, nvml_handle, image_size, batch):
    import pynvml

    torch.cuda.empty_cache()
    x = torch.randn(batch, 3, image_size, image_size, device="cuda")

    def run():
        with torch.inference_mode():
            model(x)

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for i in range(20):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
        if i >= 4 and sum(times) > 8:
            break
    latency = float(np.median(times))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    memory = int(torch.cuda.max_memory_allocated())

    samples = []
    stop_event = threading.Event()

    def read_power():
        while not stop_event.is_set():
            watts = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
            samples.append((time.perf_counter(), watts))
            time.sleep(0.01)

    worker = threading.Thread(target=read_power)
    worker.start()
    time.sleep(0.05)
    n = 0
    t0 = time.perf_counter()
    while n < 80 and (n < 3 or time.perf_counter() - t0 < 0.5):
        run()
        n += 1
    torch.cuda.synchronize()
    stop_event.set()
    worker.join()

    energy_total = 0.0
    for (t_a, p_a), (t_b, p_b) in zip(samples, samples[1:]):
        energy_total += 0.5 * (p_a + p_b) * (t_b - t_a)
    energy = energy_total / n
    del x
    return latency, memory, energy


def main():
    import pynvml

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    model = SmallCNN().cuda().eval()
    print("gpu", torch.cuda.get_device_name())
    print("torch", torch.__version__)

    base_S = [32, 64, 128, 224, 256, 384, 512]
    base_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rng = np.random.default_rng(0)
    extra_S = sorted(int(v) for v in rng.choice(
        [s for s in range(32, 513, 16) if s not in base_S], size=4, replace=False))
    extra_B = sorted(int(v) for v in rng.choice(
        [b for b in range(1, 257) if b not in base_B], size=3, replace=False))
    sizes = sorted(base_S + extra_S)
    batches = sorted(base_B + extra_B)
    print("extra S", extra_S)
    print("extra B", extra_B)

    out = Path(__file__).resolve().parent / "results" / "measurements.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["S", "B", "latency_s", "memory_bytes", "energy_j", "is_validation"]
    done = load_done(out, fields)
    if not out.exists() or out.stat().st_size == 0:
        with out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def append_row(row):
        with out.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
            f.flush()

    configs = [(s, b) for s in sizes for b in batches if (s, b) not in done]
    configs.sort(key=lambda sb: sb[0] * sb[0] * sb[1])
    print(f"осталось {len(configs)} из {len(sizes) * len(batches)}")

    for i, (image_size, batch) in enumerate(configs, 1):
        valid = int(image_size in extra_S or batch in extra_B)
        try:
            latency, memory, energy = measure_one(model, nvml_handle, image_size, batch)
            append_row({
                "S": image_size,
                "B": batch,
                "latency_s": f"{latency:.6e}",
                "memory_bytes": memory,
                "energy_j": f"{energy:.6e}",
                "is_validation": valid,
            })
            print(f"{i:3d}/{len(configs)}  S={image_size:<4} B={batch:<4}  {latency:.3e} s  {memory / 1e6:.1f} MB  {energy:.3e} J")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            append_row({
                "S": image_size,
                "B": batch,
                "latency_s": "",
                "memory_bytes": "OOM",
                "energy_j": "",
                "is_validation": valid,
            })
            print(f"{i:3d}/{len(configs)}  S={image_size:<4} B={batch:<4}  OOM")

    print("csv:", out)


if __name__ == "__main__":
    main()
