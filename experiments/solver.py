"""
Steady-state five-stage heat-balance solver for a BlueFors LD400-class
dilution refrigerator with a qubit-control wiring stack.

At each stage, the operating temperature is set by Q_dot_load(T) = Q_dot_cool(T),
where the load is the sum of (a) heat conducted in from the wiring stack
above plus (b) microwave / DC signal dissipation at the stage, and the
cooling power is the relevant pulse-tube or dilution-unit expression.

Solved top-down because each stage's temperature sets the upstream boundary
for the next-colder stage's conduction integrals. scipy.optimize.brentq is
used for each single-variable root find.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from materials import STAGE_TEMPERATURES_K
from wiring import standard_coax_line, standard_dc_bias_line, CoaxLine, DCBiasLine
from dilution_unit import DilutionUnit
from pulse_tube import PulseTube


STAGE_ORDER = ["50K_plate", "4K_plate", "still", "cold_plate", "MXC"]

# Real BlueFors still operation runs the still heater at ~10 mW to maintain
# T_still near 0.85 K independent of base load (the still is heater-controlled,
# not balance-controlled, in practice). The solver clamps T_still here and
# reports the implied heater power.
STILL_OPERATING_T_K = 0.85
HOT_OF = {
    "50K_plate": "room",
    "4K_plate": "50K_plate",
    "still": "4K_plate",
    "cold_plate": "still",
    "MXC": "cold_plate",
}


def _wiring_load_at_stage(stage, T_stages, n_coax, n_dc, drive_dBm):
    """
    Sum the heat conducted into `stage` from the next-warmer stage, plus any
    signal dissipation deposited at this stage by the attenuator chain, for a
    full wiring stack of N coax + M DC lines.
    """
    from wiring import CoaxSegment, Attenuator, CoaxLine, DCBiasLine
    from wiring import _AREA_CENTER, _AREA_OUTER, _LENGTHS

    hot = HOT_OF[stage]
    T_hot = T_stages[hot]
    T_cold = T_stages[stage]

    # Coax segment materials per pair (same scheme as standard_coax_line).
    coax_materials = {
        ("room", "50K_plate"): ("BeCu", "SS304"),
        ("50K_plate", "4K_plate"): ("BeCu", "SS304"),
        ("4K_plate", "still"): ("NbTi", "NbTi"),
        ("still", "cold_plate"): ("NbTi", "NbTi"),
        ("cold_plate", "MXC"): ("NbTi", "NbTi"),
    }
    atten_dB = {"50K_plate": 0.0, "4K_plate": 20.0, "still": 10.0,
                "cold_plate": 6.0, "MXC": 3.0}

    L = _LENGTHS[(hot, stage)]
    mat_c, mat_o = coax_materials[(hot, stage)]
    seg = CoaxSegment(mat_c, mat_o, L, _AREA_CENTER, _AREA_OUTER,
                      T_hot, T_cold, hot, stage)
    cond_per_coax = seg.conducted_heat_W()

    # Signal power that reaches this attenuator: launch power attenuated by
    # the dB sum of all upstream attenuators.
    P_in_W = 1e-3 * 10 ** (drive_dBm / 10.0)
    cumulative_dB = sum(atten_dB[s] for s in STAGE_ORDER
                        if STAGE_ORDER.index(s) < STAGE_ORDER.index(stage))
    P_at_stage_in_W = P_in_W * 10 ** (-cumulative_dB / 10.0)
    fraction = 1.0 - 10 ** (-atten_dB[stage] / 10.0)
    sig_per_coax = P_at_stage_in_W * fraction

    # DC bias lines: brass-brass twisted pair.
    A_pair = 2 * 0.0127e-6
    dc_seg = CoaxSegment("brass", "brass", L, A_pair, 0.0,
                         T_hot, T_cold, hot, stage)
    cond_per_dc = dc_seg.conducted_heat_W()

    total = n_coax * (cond_per_coax + sig_per_coax) + n_dc * cond_per_dc
    return total


def solve_steady_state(n_coax=16, n_dc=8, drive_dBm=0.0,
                       Q_parasitic_per_stage=None, verbose=False,
                       still_mode="heater_clamped"):
    """
    Top-down heat-balance solver.

    Returns
    -------
    dict with stage temperatures (K), loads per stage (W), cooling power per stage (W).
    """
    pt = PulseTube()
    du = DilutionUnit()

    T = {"room": 300.0}
    loads = {}
    cooling = {}

    parasitic = Q_parasitic_per_stage or {s: 0.0 for s in STAGE_ORDER}

    # --- 50K plate: PT stage 1 vs wiring conduction from 300 K + parasitic radiation
    def f50(T50):
        T_trial = dict(T)
        T_trial["50K_plate"] = T50
        load = _wiring_load_at_stage("50K_plate", T_trial, n_coax, n_dc, drive_dBm)
        load += parasitic["50K_plate"]
        return load - pt.Q_stage1(T50)
    T["50K_plate"] = brentq(f50, 30.1, 120.0)
    loads["50K_plate"] = _wiring_load_at_stage("50K_plate", T, n_coax, n_dc, drive_dBm) + parasitic["50K_plate"]
    cooling["50K_plate"] = pt.Q_stage1(T["50K_plate"])

    # --- 4K plate: PT stage 2 vs wiring conduction from 50K + parasitic
    def f4(T4):
        T_trial = dict(T)
        T_trial["4K_plate"] = T4
        load = _wiring_load_at_stage("4K_plate", T_trial, n_coax, n_dc, drive_dBm)
        load += parasitic["4K_plate"]
        return load - pt.Q_stage2(T4)
    T["4K_plate"] = brentq(f4, 2.51, 19.5)
    loads["4K_plate"] = _wiring_load_at_stage("4K_plate", T, n_coax, n_dc, drive_dBm) + parasitic["4K_plate"]
    cooling["4K_plate"] = pt.Q_stage2(T["4K_plate"])

    # --- Still: in real BlueFors operation T_still is heater-controlled to ~0.85 K
    # independent of base load; the still pump rate is set by the heater. Clamp
    # T_still to the operating point and report the implied heater power as
    # cooling["still"] - load (positive means the heater is supplying the excess
    # cooling power that the dilution unit alone provides at that T).
    if still_mode == "heater_clamped":
        T["still"] = STILL_OPERATING_T_K
    elif still_mode == "passive_balance":
        # Legacy: solve Q_load = Q_cool(T) without the heater.
        def fstill(Tst):
            T_trial = dict(T)
            T_trial["still"] = Tst
            load = _wiring_load_at_stage("still", T_trial, n_coax, n_dc, drive_dBm)
            load += parasitic["still"]
            return load - du.Q_still(Tst)
        T["still"] = brentq(fstill, 0.05, 1.5)
    else:
        raise ValueError("still_mode must be 'heater_clamped' or 'passive_balance'")
    loads["still"] = _wiring_load_at_stage("still", T, n_coax, n_dc, drive_dBm) + parasitic["still"]
    cooling["still"] = du.Q_still(T["still"])

    # --- Cold plate: dilution-unit CP cooling vs conduction from Still + parasitic
    def fcp(Tcp):
        T_trial = dict(T)
        T_trial["cold_plate"] = Tcp
        load = _wiring_load_at_stage("cold_plate", T_trial, n_coax, n_dc, drive_dBm)
        load += parasitic["cold_plate"]
        return load - du.Q_cold_plate(Tcp)
    T["cold_plate"] = brentq(fcp, 0.005, 0.4)
    loads["cold_plate"] = _wiring_load_at_stage("cold_plate", T, n_coax, n_dc, drive_dBm) + parasitic["cold_plate"]
    cooling["cold_plate"] = du.Q_cold_plate(T["cold_plate"])

    # --- MXC: dilution-unit MXC vs conduction from CP + parasitic
    def fmxc(Tmxc):
        T_trial = dict(T)
        T_trial["MXC"] = Tmxc
        load = _wiring_load_at_stage("MXC", T_trial, n_coax, n_dc, drive_dBm)
        load += parasitic["MXC"]
        return load - du.Q_MXC(Tmxc)
    try:
        T["MXC"] = brentq(fmxc, 0.001, 0.5)
    except ValueError:
        # No root in bracket -> MXC saturates above 500 mK
        T["MXC"] = 0.5
    loads["MXC"] = _wiring_load_at_stage("MXC", T, n_coax, n_dc, drive_dBm) + parasitic["MXC"]
    cooling["MXC"] = du.Q_MXC(T["MXC"])

    if verbose:
        print(f"  50K:  T = {T['50K_plate']:.2f} K   Q_load = {loads['50K_plate']:.3e} W   Q_cool = {cooling['50K_plate']:.3e} W")
        print(f"  4K:   T = {T['4K_plate']:.3f} K   Q_load = {loads['4K_plate']:.3e} W   Q_cool = {cooling['4K_plate']:.3e} W")
        print(f"  Still:T = {T['still']*1e3:.1f} mK   Q_load = {loads['still']*1e3:.3f} mW   Q_cool = {cooling['still']*1e3:.3f} mW")
        print(f"  CP:   T = {T['cold_plate']*1e3:.2f} mK   Q_load = {loads['cold_plate']*1e6:.1f} uW   Q_cool = {cooling['cold_plate']*1e6:.1f} uW")
        print(f"  MXC:  T = {T['MXC']*1e3:.2f} mK   Q_load = {loads['MXC']*1e6:.3f} uW   Q_cool = {cooling['MXC']*1e6:.3f} uW")

    return {"T": T, "loads": loads, "cooling": cooling}


def main():
    # Reasonable parasitic radiation + structural conduction baseline (catalog-like).
    parasitic = {
        "50K_plate": 5.0,       # ~5 W MLI/radiation/structural to 50K
        "4K_plate": 50e-3,      # ~50 mW radiation through 50K shield to 4K
        "still": 5e-3,          # ~5 mW residual still load
        "cold_plate": 50e-6,    # ~50 uW residual CP load
        "MXC": 1e-6,            # ~1 uW residual MXC load (film flow, joints)
    }

    print("=" * 70)
    print("CASE A: empty fridge (no wiring stack, baseline parasitic only)")
    print("=" * 70)
    res = solve_steady_state(n_coax=0, n_dc=0, drive_dBm=-100.0,
                             Q_parasitic_per_stage=parasitic, verbose=True)
    print()

    for n_coax, n_dc, dBm in [(16, 8, 0.0), (32, 16, 0.0), (64, 32, 0.0),
                              (16, 8, 10.0), (16, 8, 20.0)]:
        print("=" * 70)
        print(f"CASE: {n_coax} coax + {n_dc} DC at +{dBm} dBm drive (with parasitic)")
        print("=" * 70)
        solve_steady_state(n_coax=n_coax, n_dc=n_dc, drive_dBm=dBm,
                           Q_parasitic_per_stage=parasitic, verbose=True)
        print()


if __name__ == "__main__":
    main()
