"""
Thermal conductivity integrals for common cryogenic materials.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_PATH = os.path.normpath(os.path.join(_HERE, "..", "data", "kappa_tables.csv"))

STAGE_TEMPERATURES_K = {
    "room": 300.0,
    "50K_plate": 50.0,
    "4K_plate": 4.0,
    "still": 0.8,
    "cold_plate": 0.1,
    "MXC": 0.01,
}

STAGE_PAIRS = [
    ("room", "50K_plate"),
    ("50K_plate", "4K_plate"),
    ("4K_plate", "still"),
    ("still", "cold_plate"),
    ("cold_plate", "MXC"),
]


def _load_table():
    return pd.read_csv(_TABLE_PATH, comment="#")


@lru_cache(maxsize=None)
def _table_arrays(material):
    df = _load_table()
    if material not in df.columns:
        avail = [c for c in df.columns if c != "T_K"]
        raise ValueError("Unknown material {!r}. Available: {}".format(material, avail))
    T = df["T_K"].to_numpy(dtype=float)
    k = df[material].to_numpy(dtype=float)
    order = np.argsort(T)
    return T[order], k[order]


def kappa(material, T_K):
    """Thermal conductivity kappa(T) in W/m/K via log-log linear interpolation."""
    T_arr, k_arr = _table_arrays(material)
    logT = np.log(T_arr)
    logK = np.log(k_arr)
    if T_K <= T_arr[0]:
        n = (logK[1] - logK[0]) / (logT[1] - logT[0])
        return float(k_arr[0] * (T_K / T_arr[0]) ** n)
    if T_K >= T_arr[-1]:
        n = (logK[-1] - logK[-2]) / (logT[-1] - logT[-2])
        return float(k_arr[-1] * (T_K / T_arr[-1]) ** n)
    return float(np.exp(np.interp(np.log(T_K), logT, logK)))


def _segment_integral(T_a, T_b, k_a, k_b):
    """Closed-form integral of kappa(T) = k_a * (T/T_a)^n between T_a and T_b > T_a."""
    if k_a <= 0 or k_b <= 0:
        return 0.5 * (k_a + k_b) * (T_b - T_a)
    n = (np.log(k_b) - np.log(k_a)) / (np.log(T_b) - np.log(T_a))
    if abs(n + 1.0) < 1e-9:
        return k_a * T_a * np.log(T_b / T_a)
    coef = k_a / (T_a ** n) / (n + 1.0)
    return coef * (T_b ** (n + 1) - T_a ** (n + 1))


def conduction_integral(material, T_hot, T_cold):
    """Integral of kappa(T) from T_cold to T_hot in W/m."""
    if T_hot < T_cold:
        T_hot, T_cold = T_cold, T_hot
    T_arr, _ = _table_arrays(material)
    interior = [T for T in T_arr if T_cold < T < T_hot]
    nodes_T = np.array([T_cold] + interior + [T_hot])
    nodes_k = np.array([kappa(material, T) for T in nodes_T])
    total = 0.0
    for i in range(len(nodes_T) - 1):
        total += _segment_integral(nodes_T[i], nodes_T[i + 1], nodes_k[i], nodes_k[i + 1])
    return total


def list_materials():
    df = _load_table()
    return [c for c in df.columns if c != "T_K"]


def main():
    mats = list_materials()
    print("Loaded kappa(T) table from", _TABLE_PATH)
    print("Materials:", ", ".join(mats))
    print()
    header = "{:<22}".format("pair") + "".join("{:>12}".format(m) for m in mats)
    print(header)
    print("-" * len(header))
    for hot_name, cold_name in STAGE_PAIRS:
        T_hot = STAGE_TEMPERATURES_K[hot_name]
        T_cold = STAGE_TEMPERATURES_K[cold_name]
        label = "{:>5.1f}K -> {:<5.2f}K".format(T_hot, T_cold)
        row = [conduction_integral(m, T_hot, T_cold) for m in mats]
        print("{:<22}".format(label) + "".join("{:>12.3e}".format(v) for v in row))
    print()
    print("Sanity checks (SS304):")
    print("  300 K -> 4 K:    {:.3e} W/m  (reference ~3.06e3)".format(conduction_integral('SS304', 300, 4)))
    print("  4 K -> 0.1 K:    {:.3e} W/m  (reference ~0.34)".format(conduction_integral('SS304', 4, 0.1)))


if __name__ == "__main__":
    main()
