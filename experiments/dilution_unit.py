"""
Cooling-power model for a 3He/4He dilution refrigerator.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# Pobell Eq. 7.30 enthalpy coefficients (J / (mol * K^2)).
H_DILUTE_COEF = 95.0
H_CONCENTRATED_COEF = 11.0
ENTHALPY_DELTA = H_DILUTE_COEF - H_CONCENTRATED_COEF  # 84.0

# Latent heat of vaporization of 3He at the still (~0.7 K), J/mol.
L3_VAP = 25.0


@dataclass
class DilutionUnit:
   published: ~30 mW.

    def Q_MXC(self, T_MXC_K):
        """Mixing-chamber cooling power at temperature T_MXC_K (W)."""
        T_in = self.T_in_factor * T_MXC_K
        # Q = n_dot * (H_dilute(T_MXC) - H_concentrated(T_in))
        #   = n_dot * (95*T_MXC^2 - 11*T_in^2)
        H_dilute = H_DILUTE_COEF * T_MXC_K ** 2
        H_concentrated_in = H_CONCENTRATED_COEF * T_in ** 2
        return self.n_dot_3 * max(H_dilute - H_concentrated_in, 0.0)

    def T_MXC_at_load(self, Q_load_W):
        """Inverse: given a steady heat load at the MXC, what is T_MXC?"""
        # Q = n_dot * (95 - 11 * T_in_factor^2) * T_MXC^2
        coef = self.n_dot_3 * (H_DILUTE_COEF - H_CONCENTRATED_COEF * self.T_in_factor ** 2)
        if coef <= 0:
            raise ValueError("Effective dilution coefficient is non-positive; check T_in_factor.")
        return float(np.sqrt(max(Q_load_W, 0.0) / coef))

    def Q_still(self, T_still_K):
        return self.Q_still_nominal_W * (T_still_K / 0.8)

    def Q_cold_plate(self, T_cp_K):
        """
        Cold-plate cooling power. Anchored to the LD400 spec at 100 mK and
        scaled as T^2 (same dilution-style scaling, since CP cooling is
        provided by the dilute return stream's enthalpy).
        """
        return self.Q_cp_100mK_W * (T_cp_K / 0.1) ** 2

    def T_cp_at_load(self, Q_load_W):
        return 0.1 * float(np.sqrt(max(Q_load_W, 0.0) / self.Q_cp_100mK_W))


def main():
    du = DilutionUnit()
    print("Dilution-unit model parameters:")
    print(f"  n_dot_3 = {du.n_dot_3 * 1e6:.1f} umol/s")
    print(f"  T_in_factor = {du.T_in_factor}")
    print(f"  Q_cp(100 mK) reference = {du.Q_cp_100mK_W * 1e6:.1f} uW")
    print(f"  Q_still nominal = {du.Q_still_nominal_W * 1e3:.1f} mW")
    print()

    print("MXC cooling power vs T_MXC:")
    for T_mK in [10, 15, 20, 30, 50, 100, 200, 500]:
        T = T_mK / 1000.0
        print(f"  T_MXC = {T_mK:>4d} mK -> Q = {du.Q_MXC(T) * 1e6:>9.3f} uW")
    print()

    print("T_MXC reached at various heat loads:")
    for Q_uW in [1, 5, 10, 14, 25, 50, 100]:
        T = du.T_MXC_at_load(Q_uW * 1e-6)
        print(f"  Q_load = {Q_uW:>4d} uW -> T_MXC = {T * 1e3:>6.2f} mK")
    print()

    print("Cold-plate cooling power vs T_CP:")
    for T_mK in [50, 80, 100, 150, 200]:
        T = T_mK / 1000.0
        print(f"  T_CP = {T_mK:>4d} mK -> Q = {du.Q_cold_plate(T) * 1e6:>9.3f} uW")
    print()

    print("Still cooling power vs T_still:")
    for T_mK in [500, 700, 800, 900]:
        T = T_mK / 1000.0
        print(f"  T_still = {T_mK:>4d} mK -> Q = {du.Q_still(T) * 1e3:>6.3f} mW")


if __name__ == "__main__":
    main()
