"""
Cooling-power model for a 3He/4He dilution refrigerator.

The mixing-chamber cooling power, for a continuous-circulation DR running at
3He molar flow rate n_dot (mol/s), is

    Q_dot_MXC(T_MXC, T_in) = n_dot * (H_out(T_MXC) - H_in(T_in))            [W]

where H is the enthalpy of the 3He stream above the absolute zero reference.
At low T (Pobell Eq. 7.30), the enthalpies reduce to

    H_dilute(T)   ~ 95.0 * T**2  J/mol      (3He in dilute phase)
    H_concentrated(T) ~ 11.0 * T**2 J/mol   (3He in concentrated phase, post-HX)

so the standard textbook result is

    Q_dot_MXC ~ n_dot * (95 - 11) * T_MXC**2 - n_dot * 11 * (T_in**2 - T_MXC**2)
              ~ 84 * n_dot * (T_MXC**2 - T_in**2)                            [W]

with T_in the 3He temperature at the inlet of the mixing chamber (i.e., after
the final heat exchanger between the still return and the incoming concentrated
stream). Below the T_in << T_MXC limit, Q_MXC ~ 84 * n_dot * T_MXC**2.

The still cooling power is set by the 4He evaporative cooling at ~0.7-0.9 K:
the still pump extracts mostly 3He vapor, but the residual 4He carries the
latent heat of vaporization L_4 ~ 23.6 J/mol. For a 3He-dominated still flow,
the still cooling power is approximately

    Q_dot_still ~ n_dot * L_3       L_3 ~ 25 J/mol latent enthalpy at ~0.7 K

In a practical LD400, n_dot_3 is ~500 umol/s, giving Q_still ~ 12-30 mW
matching the catalog value.

The cold-plate stage is set by the heat-exchanger thermal anchor between the
still return and the dilute-stream return; the published LD400 specs treat it
as ~700 uW at 100 mK and we use that as a tabulated value.
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
    """
    Closed-form Pobell dilution-unit cooling-power model.

    Parameters
    ----------
    n_dot_3 : float
        3He molar circulation rate (mol/s). LD400 default: ~500e-6.
    T_in_K : float
        3He temperature at the mixing-chamber inlet (K), set by the last
        heat exchanger before the MXC. LD400-class HX achieves T_in ~ 1.5 * T_MXC.
        For a steady-state model we treat T_in as proportional to T_MXC and
        absorb the proportionality into the effective enthalpy coefficient.
    Q_cp_100mK : float
        Reference cold-plate cooling power at 100 mK (W). LD400 published: 700 uW.
    Q_still_nominal : float
        Reference still cooling power at the operating point (W).
        LD400 published: ~30 mW.
    """

    n_dot_3: float = 500e-6
    T_in_factor: float = 1.5  # T_in_K = T_in_factor * T_MXC at HX equilibrium
    Q_cp_100mK_W: float = 700e-6
    Q_still_nominal_W: float = 30e-3

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
        """
        Still cooling power. Pobell models this as n_dot * L_3 at the operating
        still temperature; LD400 quotes ~30 mW at the operating point. Use a
        gentle linear interpolation around the catalog value as T_still varies.
        """
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
