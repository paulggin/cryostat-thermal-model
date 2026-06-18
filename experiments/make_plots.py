"""
Generate the four headline plots for the Cryostat Thermal Model project:

  1. kappa(T) for each tabulated material (log-log).
  2. Heat-load breakdown per stage: conducted vs signal dissipation, for
     N_coax + N_dc nominal stack.
  3. Dilution-unit cooling-power curve overlaid with operating points for
     a sweep of N_coax counts.
  4. T_MXC vs N_coax sweep, with the 50 mK qubit-thermal threshold marked.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os

from materials import list_materials, kappa, STAGE_TEMPERATURES_K
from wiring import standard_coax_line, standard_dc_bias_line
from dilution_unit import DilutionUnit
from solver import solve_steady_state

PLOTS = "/sessions/lucid-elegant-davinci/mnt/Cowork Brainstem/Quantum Job Search/Portfolio/Cryostat_Thermal_Model/outputs/plots"
os.makedirs(PLOTS, exist_ok=True)

PARASITIC = {
    "50K_plate": 35.0,
    "4K_plate": 1.3,
    "still": 28e-3,
    "cold_plate": 600e-6,
    "MXC": 10e-6,
}


# ----------------------------------------------------------------------
# Plot 1: kappa(T) library
# ----------------------------------------------------------------------
def plot_kappa():
    T_grid = np.logspace(np.log10(0.01), np.log10(300), 400)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for m in list_materials():
        k_arr = np.array([kappa(m, T) for T in T_grid])
        ax.loglog(T_grid, k_arr, label=m)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"Thermal conductivity $\kappa$ [W/m/K]")
    ax.set_title("Cryogenic thermal conductivity library")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    for s in ["50K_plate", "4K_plate", "still", "cold_plate", "MXC"]:
        ax.axvline(STAGE_TEMPERATURES_K[s], color="grey", lw=0.6, alpha=0.4)
    fig.tight_layout()
    out = os.path.join(PLOTS, "kappa_vs_T.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------
# Plot 2: Heat-load breakdown per stage for 16 coax + 8 DC, +0 dBm
# ----------------------------------------------------------------------
def plot_load_breakdown():
    stages = ["50K_plate", "4K_plate", "still", "cold_plate", "MXC"]
    n_coax, n_dc, dBm = 16, 8, 0.0

    coax = standard_coax_line(drive_power_dBm=dBm)
    dc = standard_dc_bias_line()
    coax_cond = coax.conducted_heat_by_stage()
    coax_sig = coax.signal_dissipation_by_stage()
    dc_cond = dc.conducted_heat_by_stage()

    conducted_W = np.array([n_coax * coax_cond[s] + n_dc * dc_cond[s] for s in stages])
    signal_W = np.array([n_coax * coax_sig[s] for s in stages])
    parasitic_W = np.array([PARASITIC[s] for s in stages])

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(x, parasitic_W, label="Parasitic radiation / structural")
    ax.bar(x, conducted_W, bottom=parasitic_W,
           label=f"Conducted ({n_coax} coax + {n_dc} DC)")
    ax.bar(x, signal_W, bottom=parasitic_W + conducted_W,
           label=f"Signal dissipation ({n_coax} coax @ +{dBm:.0f} dBm)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["50 K", "4 K", "Still", "Cold plate", "MXC"], rotation=0)
    ax.set_ylabel("Heat load [W]")
    ax.set_title(f"Per-stage heat-load decomposition ({n_coax} coax + {n_dc} DC, +{dBm:.0f} dBm)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(PLOTS, "heat_load_breakdown.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------
# Plot 3: Dilution-unit Q_MXC(T) curve with operating points overlaid
# ----------------------------------------------------------------------
def plot_dilution_curve():
    du = DilutionUnit()
    T_grid = np.linspace(0.005, 0.15, 400)
    Q_grid = np.array([du.Q_MXC(T) for T in T_grid])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(T_grid * 1e3, Q_grid * 1e6, "k-", lw=1.5,
            label=r"Pobell: $Q_{MXC} \propto T_{MXC}^2$")
    op_T = []
    op_Q = []
    op_N = []
    for N in [0, 16, 32, 64, 128, 256, 512]:
        ndc = max(N // 2, 1) if N > 0 else 0
        r = solve_steady_state(n_coax=N, n_dc=ndc, drive_dBm=0.0,
                               Q_parasitic_per_stage=PARASITIC)
        op_T.append(r["T"]["MXC"])
        op_Q.append(r["loads"]["MXC"])
        op_N.append(N)
    op_T = np.array(op_T)
    op_Q = np.array(op_Q)
    ax.scatter(op_T * 1e3, op_Q * 1e6, c="C3", zorder=5)
    for n, tT, qQ in zip(op_N, op_T, op_Q):
        ax.annotate(f"  N={n}", (tT * 1e3, qQ * 1e6), fontsize=8, color="C3")
    ax.axhline(14.0, color="grey", lw=0.7, ls="--",
               label="LD400 catalog spec: 14 uW at 20 mK")
    ax.set_xlabel("Mixing chamber temperature [mK]")
    ax.set_ylabel(r"Cooling power $\dot{Q}_{MXC}$ [$\mu$W]")
    ax.set_title("Dilution-unit cooling-power curve with operating points")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 280)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(PLOTS, "dilution_curve_with_op_points.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------
# Plot 4: T_MXC vs N_coax sweep with 50 mK qubit-thermal threshold
# ----------------------------------------------------------------------
def plot_sweep():
    Ns = np.array([0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    T_MXC = []
    for N in Ns:
        ndc = max(N // 2, 1) if N > 0 else 0
        r = solve_steady_state(n_coax=N, n_dc=ndc, drive_dBm=0.0,
                               Q_parasitic_per_stage=PARASITIC)
        T_MXC.append(r["T"]["MXC"])
    T_MXC = np.array(T_MXC) * 1e3  # to mK

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(Ns, T_MXC, "o-", color="C0", lw=1.4, label="Model")
    ax.axhline(50.0, color="C3", ls="--",
               label="50 mK qubit-thermal threshold")
    ax.axhline(20.0, color="grey", ls=":",
               label="LD400 published 20 mK operating point")
    ax.set_xlabel("Number of coax control lines (DC lines = N_coax / 2)")
    ax.set_ylabel("Mixing chamber temperature [mK]")
    ax.set_title("T_MXC vs wiring stack size, +0 dBm drive")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS, "T_MXC_vs_N_coax.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def main():
    plot_kappa()
    plot_load_breakdown()
    plot_dilution_curve()
    plot_sweep()


if __name__ == "__main__":
    main()
