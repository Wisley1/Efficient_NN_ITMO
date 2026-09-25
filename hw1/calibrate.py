import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from equations import bytes_moved, energy, flops, latency, memory


def mape(pred, y):
    return float(np.mean(np.abs(pred - y) / y))


def load_measurements(csv_path):
    text = Path(csv_path).read_text().strip().splitlines()
    fields = ["S", "B", "latency_s", "memory_bytes", "energy_j", "is_validation"]
    if text[0].startswith("S"):
        rows = list(csv.DictReader(text))
    else:
        rows = list(csv.DictReader(text, fieldnames=fields))
    data = [r for r in rows if r["memory_bytes"] != "OOM"]
    return {
        "S": np.array([float(r["S"]) for r in data]),
        "B": np.array([float(r["B"]) for r in data]),
        "lat": np.array([float(r["latency_s"]) for r in data]),
        "mem": np.array([float(r["memory_bytes"]) for r in data]),
        "en": np.array([float(r["energy_j"]) for r in data]),
        "val": np.array([int(r["is_validation"]) for r in data], dtype=bool),
    }


def fit(csv_path, out_dir):
    measured = load_measurements(csv_path)
    S = measured["S"]
    B = measured["B"]
    lat = measured["lat"]
    mem = measured["mem"]
    en = measured["en"]
    val = measured["val"]
    train = ~val

    y_lat = lat[train]
    lat_fit = least_squares(
        lambda z: (latency(S[train], B[train], np.exp(z)) - y_lat) / y_lat,
        np.log([6e-4, 2e12, 2e11]),
        method="trf",
    )
    theta_lat = tuple(float(v) for v in np.exp(lat_fit.x))

    y_en = en[train]
    en_fit = least_squares(
        lambda z: (
            energy(S[train], B[train], (np.exp(z[0]), np.exp(z[1]), np.exp(z[2]), *theta_lat))
            - y_en
        ) / y_en,
        np.log([70.0, 1e-12, 1e-12]),
        method="trf",
    )
    theta_en = (
        float(np.exp(en_fit.x[0])),
        float(np.exp(en_fit.x[1])),
        float(np.exp(en_fit.x[2])),
        *theta_lat,
    )

    lat_hat = latency(S, B, theta_lat)
    en_hat = energy(S, B, theta_en)
    mem_hat = memory(S, B)

    t_launch = np.full_like(lat, theta_lat[0])
    t_comp = flops(S, B) / theta_lat[1]
    t_mem = bytes_moved(S, B) / theta_lat[2]
    winner = np.argmax(np.vstack([t_launch, t_mem, t_comp]), axis=0)

    theta = {
        "latency": {
            "launch_s": theta_lat[0],
            "compute_flops_per_s": theta_lat[1],
            "bandwidth_bytes_per_s": theta_lat[2],
        },
        "energy": {
            "static_w": theta_en[0],
            "j_per_flop": theta_en[1],
            "j_per_byte": theta_en[2],
        },
        "mape": {
            "latency_train": mape(lat_hat[train], lat[train]),
            "latency_val": mape(lat_hat[val], lat[val]),
            "energy_train": mape(en_hat[train], en[train]),
            "energy_val": mape(en_hat[val], en[val]),
            "memory_all": mape(mem_hat, mem),
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "theta.json").write_text(json.dumps(theta, indent=2))
    return theta, winner, train, val


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    theta, winner, train, val = fit(root / "results" / "measurements.csv", root / "results")
    print("launch_s", theta["latency"]["launch_s"])
    print("compute TFLOP/s", theta["latency"]["compute_flops_per_s"] / 1e12)
    print("bandwidth GB/s", theta["latency"]["bandwidth_bytes_per_s"] / 1e9)
    print("static W", theta["energy"]["static_w"])
    print("MAPE latency train/val", theta["mape"]["latency_train"], theta["mape"]["latency_val"])
    print("MAPE energy  train/val", theta["mape"]["energy_train"], theta["mape"]["energy_val"])
    print("MAPE memory all", theta["mape"]["memory_all"])
    for i, name in enumerate(["launch", "memory", "compute"]):
        print(
            f"regime {name}: train {(winner[train] == i).mean():.2f}  val {(winner[val] == i).mean():.2f}"
        )
    print("saved", root / "results" / "theta.json")
