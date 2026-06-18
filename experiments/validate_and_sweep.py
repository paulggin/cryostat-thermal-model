"""
Validation + N_coax sweep against BlueFors LD400 published specifications.

BlueFors LD400 catalog:
  Stage 1 (50K plate):  ~40 W cooling capacity at 45 K
  Stage 2 (4K plate):   ~1.5 W cooling capacity at 4.2 K
  Still (~0.8 K):       ~30 mW cooling capacity
  Cold plate (~0.1 K):  ~700 uW cooling capacity
  MXC (~10 mK):         ~14 uW cooling capacity, base temperature < 10 mK

"""

from __future__ import annotations

import numpy as np

from solver import solve_steady_state


CATALOG = {
    "50K_plate": {"T_ref_K": 45.0, "Q_W": 40.0},
    "4K_plate":  {"T_ref_K": 4.2,  "Q_W": 1.5},
    "still":     {"T_ref_K": 0.8,  "Q_W": 30e-3},
    "cold_plate":{"T_ref_K": 0.1,  "Q_W": 700e-6},
    "MXC":       {"T_ref_K": 0.01, "Q_W": 14e-6},
}

# Catalog-magnitude parasitic loads (chosen to bring each stage near its
# nominal operating point in the absence of any wiring).
PARASITIC_CATALOG = {
    "50K_plate": 35.0,        # 35 W -> T_50 lands near 45 K
    "4K_plate": 1.3,          # 1.3 W -> T_4 lands near 4.2 K
    "still": 28e-3,           # 28 mW -> T_still near 800 mK
    "cold_plate": 600e-6,     # 600 uW -> T_CP near 100 mK
    "MXC": 10e-6,             # 10 uW -> T_MXC near 20 mK
}


def main():
    print("=" * 78)
    print("VALIDATION: catalog-parasitic-only fridge vs LD400 published spec")
    print("=" * 78)
    res = solve_steady_state(n_coax=0, n_dc=0, drive_dBm=-100.0,
                             Q_parasitic_per_stage=PARASITIC_CATALOG)
    print(f"{'stage':<12} {'model T':>12} {'catalog T':>12} {'model Q':>14} {'catalog Q':>14}")
    print("-" * 78)
    for s in ["50K_plate", "4K_plate", "still", "cold_plate", "MXC"]:
        T_model = res["T"][s]
        Q_model = res["loads"][s]
        T_cat = CATALOG[s]["T_ref_K"]
        Q_cat = CATALOG[s]["Q_W"]
        if T_cat < 1:
            print(f"{s:<12} {T_model*1e3:>9.2f} mK {T_cat*1e3:>9.2f} mK "
                  f"{Q_model*1e6:>11.1f} uW {Q_cat*1e6:>11.1f} uW")
        else:
            print(f"{s:<12} {T_model:>10.2f} K {T_cat:>10.2f} K "
                  f"{Q_model:>12.3e} W {Q_cat:>12.3e} W")
    print()

    print("=" * 78)
    print("N_coax SWEEP at +0 dBm drive, N_dc = N_coax/2, full parasitic baseline")
    print("=" * 78)
    print(f"{'N_coax':>8} {'N_dc':>6} {'T_50K':>10} {'T_4K':>10} {'T_still':>10} {'T_CP':>10} {'T_MXC':>10}")
    print("-" * 78)
    rows = []
    for N in [0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
        ndc = max(N // 2, 1) if N > 0 else 0
        r = solve_steady_state(n_coax=N, n_dc=ndc, drive_dBm=0.0,
                               Q_parasitic_per_stage=PARASITIC_CATALOG)
        T = r["T"]
        print(f"{N:>8d} {ndc:>6d} "
              f"{T['50K_plate']:>8.2f} K "
              f"{T['4K_plate']:>8.3f} K "
              f"{T['still']*1e3:>7.1f} mK "
              f"{T['cold_plate']*1e3:>7.2f} mK "
              f"{T['MXC']*1e3:>7.2f} mK")
        rows.append((N, ndc, T['50K_plate'], T['4K_plate'], T['still'],
                     T['cold_plate'], T['MXC']))
    print()

    # Find the N_coax at which T_MXC crosses 50 mK (engineering threshold for
    # transmon qubit operation: typical f_01 ~ 5 GHz corresponds to hf/k_B ~ 240 mK,
    # but qubit thermal-population limit demands T < ~50 mK to keep n_thermal < 1%).
    Ns = np.array([r[0] for r in rows])
    Ts_MXC = np.array([r[6] for r in rows])
    if (Ts_MXC > 0.050).any() and (Ts_MXC <= 0.050).any():
        i = np.argmax(Ts_MXC > 0.050)
        if i > 0:
            N_lo, N_hi = Ns[i-1], Ns[i]
            T_lo, T_hi = Ts_MXC[i-1], Ts_MXC[i]
            N_50mK = N_lo + (0.050 - T_lo) * (N_hi - N_lo) / (T_hi - T_lo)
            print(f"Engineering limit (interp): N_coax = {N_50mK:.0f} brings T_MXC to 50 mK")
    np.savetxt(r"C:\Users\Paul\OneDrive\Desktop\Portfolio Projects\Cryostat - Thermal Model\outputs\data\sweep_T_MXC_vs_N_coax.csv",
               np.array(rows),
               header="N_coax,N_dc,T_50K_K,T_4K_K,T_still_K,T_CP_K,T_MXC_K",
               delimiter=",", comments="")


if __name__ == "__main__":
    main()
