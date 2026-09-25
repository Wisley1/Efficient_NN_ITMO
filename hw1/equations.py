import numpy as np


def flops(image_size, batch):
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return b * (17712.0 * s**2 + 313700.0)


def memory(image_size, batch):
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return 4161296.0 + 44.0 * b * s**2


def bytes_moved(image_size, batch):
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    s2 = s**2
    terms = [
        4.0 * (b * 3 * s2 + 4704 + b * 8 * s2),
        4.0 * (2 * b * 8 * s2),
        4.0 * (b * 8 * s2 + b * 2 * s2),
        4.0 * (b * 2 * s2 + 51200 + b * 4 * s2),
        4.0 * (2 * b * 4 * s2),
        4.0 * (b * 4 * s2 + 73728 + b * 2 * s2),
        4.0 * (2 * b * 2 * s2),
        4.0 * (b * 2 * s2 + 32768 + b * 4 * s2),
        4.0 * (2 * b * 4 * s2),
        4.0 * (b * 4 * s2 + 589824 + b * s2),
        4.0 * (2 * b * s2),
        4.0 * (b * s2 + 131072 + b * 2 * s2),
        4.0 * (2 * b * 2 * s2),
        4.0 * (b * 2 * s2 + b * 512),
        4.0 * (b * 512 + 131328 + b * 256),
        4.0 * (2 * b * 256),
        4.0 * (b * 256 + 25700 + b * 100),
    ]
    return sum(terms)


def latency(image_size, batch, theta):
    launch, compute, bandwidth = theta
    return launch + np.maximum(
        flops(image_size, batch) / compute,
        bytes_moved(image_size, batch) / bandwidth,
    )


def energy(image_size, batch, theta_energy):
    static, j_flop, j_byte, launch, compute, bandwidth = theta_energy
    t = latency(image_size, batch, (launch, compute, bandwidth))
    return (
        static * t
        + j_flop * flops(image_size, batch)
        + j_byte * bytes_moved(image_size, batch)
    )
