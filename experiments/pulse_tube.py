"""
Cryomech PT415-class two-stage pulse-tube cryocooler cooling-power model.

  Stage 1 (~50 K anchor):  0 W @ 30 K, 40 W @ 45 K, 60 W @ 80 K  (approx)
  Stage 2 (~4 K anchor):   0 W @ 2.5 K, 1.5 W @ 4.2 K, 3 W @ 10 K  (approx)

"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# Cryomech PT415 catalog values, transcribed from the datasheet (W vs K).
_PT415_STAGE1 = np.array([
    [30.0, 0.0],
    [40.0, 25.0],
    [45.0, 40.0],
    [55.0, 50.0],
    [80.0, 65.0],
    [120.0, 80.0],
])

_PT415_STAGE2 = np.array([
    [2.5, 0.0],
    [3.5, 0.7],
    [4.2, 1.5],
    [6.0, 2.4],
    [10.0, 3.0],
    [20.0, 4.0],
])


@dataclass
class PulseTube:
    stage1_table: np.ndarray = None
    stage2_table: np.ndarray = None

    def __post_init__(self):
        if self.stage1_table is None:
            self.stage1_table = _PT415_STAGE1.copy()
        if self.stage2_table is None:
            self.stage2_table = _PT415_STAGE2.copy()

    def Q_stage1(self, T_K):
        T, Q = self.stage1_table[:, 0], self.stage1_table[:, 1]
        return float(np.clip(np.interp(T_K, T, Q), 0.0, None))

    def Q_stage2(self, T_K):
        T, Q = self.stage2_table[:, 0], self.stage2_table[:, 1]
        return float(np.clip(np.interp(T_K, T, Q), 0.0, None))


def main():
    pt = PulseTube()
    print("Cryomech PT415 cooling-power curves (catalog values):")
    print()
    print("Stage 1 (~50 K anchor):")
    for T in [30, 35, 40, 45, 50, 55, 60, 70, 80, 100]:
        print(f"  T = {T:>4d} K -> Q = {pt.Q_stage1(T):>6.2f} W")
    print()
    print("Stage 2 (~4 K anchor):")
    for T_tenth in [25, 30, 35, 42, 50, 60, 80, 100, 150, 200]:
        T = T_tenth / 10.0
        print(f"  T = {T:>5.1f} K -> Q = {pt.Q_stage2(T):>6.2f} W")


if __name__ == "__main__":
    main()
