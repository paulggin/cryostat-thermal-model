from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from materials import conduction_integral, STAGE_TEMPERATURES_K

# Standard BlueFors LD400 stage names and temperatures (K).
STAGES = ["room", "50K_plate", "4K_plate", "still", "cold_plate", "MXC"]
STAGE_T = STAGE_TEMPERATURES_K


@dataclass
class CoaxSegment:
    """One length of coax between two adjacent stages."""
    material_center: str
    material_outer: str
    length_m: float
    area_center_m2: float
    area_outer_m2: float
    T_hot_K: float
    T_cold_K: float
    stage_hot: str
    stage_cold: str

    def conducted_heat_W(self) -> float:
        K_c = conduction_integral(self.material_center, self.T_hot_K, self.T_cold_K)
        K_o = conduction_integral(self.material_outer, self.T_hot_K, self.T_cold_K)
        return (self.area_center_m2 * K_c + self.area_outer_m2 * K_o) / self.length_m


@dataclass
class Attenuator:
    """A fixed-loss attenuator anchored to one stage."""
    stage: str
    attenuation_dB: float


@dataclass
class CoaxLine:
    """A qubit-control line: series of segments + attenuator chain."""
    segments: List[CoaxSegment]
    attenuators: List[Attenuator]
    drive_power_dBm: float = 0.0

    def conducted_heat_by_stage(self) -> Dict[str, float]:
        """Heat conducted into each cold stage from the segment above it."""
        loads = {s: 0.0 for s in STAGES}
        for seg in self.segments:
            loads[seg.stage_cold] += seg.conducted_heat_W()
        return loads

    def signal_dissipation_by_stage(self) -> Dict[str, float]:
        """Microwave drive power dissipated by attenuators at each stage."""
        loads = {s: 0.0 for s in STAGES}
        P_in_W = 1e-3 * 10 ** (self.drive_power_dBm / 10.0)
        for atten in self.attenuators:
            fraction_absorbed = 1.0 - 10 ** (-atten.attenuation_dB / 10.0)
            loads[atten.stage] += P_in_W * fraction_absorbed
            P_in_W *= 10 ** (-atten.attenuation_dB / 10.0)
        return loads

    def total_heat_by_stage(self) -> Dict[str, float]:
        c = self.conducted_heat_by_stage()
        s = self.signal_dissipation_by_stage()
        return {k: c[k] + s[k] for k in STAGES}


@dataclass
class DCBiasLine:
    """A DC bias line: series of phosphor-bronze (brass-like) segments."""
    segments: List[CoaxSegment]

    def conducted_heat_by_stage(self) -> Dict[str, float]:
        loads = {s: 0.0 for s in STAGES}
        for seg in self.segments:
            loads[seg.stage_cold] += seg.conducted_heat_W()
        return loads

    def total_heat_by_stage(self) -> Dict[str, float]:
        return self.conducted_heat_by_stage()


# ----------------------------------------------------------------------
# Standard default geometries.
# ----------------------------------------------------------------------

# UT-085-class semi-rigid coax cross-sections (m^2). OD 2.20 mm, ID 0.51 mm,
# dielectric PTFE. Center conductor solid 0.51 mm dia; outer wall thickness ~0.20 mm.
_AREA_CENTER = np.pi * (0.51e-3 / 2.0) ** 2  # 2.04e-7 m^2
_AREA_OUTER = np.pi * ((2.20e-3 / 2.0) ** 2 - (1.80e-3 / 2.0) ** 2)  # ~1.26e-6 m^2

# Standard inter-stage lengths (m) for a BlueFors LD400.
_LENGTHS = {
    ("room", "50K_plate"): 0.220,
    ("50K_plate", "4K_plate"): 0.220,
    ("4K_plate", "still"): 0.220,
    ("still", "cold_plate"): 0.100,
    ("cold_plate", "MXC"): 0.100,
}


def standard_coax_line(drive_power_dBm: float = 0.0) -> CoaxLine:
    pairs = [
        ("room", "50K_plate", "BeCu", "SS304"),
        ("50K_plate", "4K_plate", "BeCu", "SS304"),
        ("4K_plate", "still", "NbTi", "NbTi"),
        ("still", "cold_plate", "NbTi", "NbTi"),
        ("cold_plate", "MXC", "NbTi", "NbTi"),
    ]
    segments = [
        CoaxSegment(
            material_center=center,
            material_outer=outer,
            length_m=_LENGTHS[(hot, cold)],
            area_center_m2=_AREA_CENTER,
            area_outer_m2=_AREA_OUTER,
            T_hot_K=STAGE_T[hot],
            T_cold_K=STAGE_T[cold],
            stage_hot=hot,
            stage_cold=cold,
        )
        for hot, cold, center, outer in pairs
    ]
    attenuators = [
        Attenuator("50K_plate", 0.0),
        Attenuator("4K_plate", 20.0),
        Attenuator("still", 10.0),
        Attenuator("cold_plate", 6.0),
        Attenuator("MXC", 3.0),
    ]
    return CoaxLine(segments=segments, attenuators=attenuators, drive_power_dBm=drive_power_dBm)


def standard_dc_bias_line() -> DCBiasLine:
    A_pair = 2 * 0.0127e-6  # m^2 (two wires)
    segments = []
    for (hot, cold), L in _LENGTHS.items():
        segments.append(
            CoaxSegment(
                material_center="brass",
                material_outer="brass",
                length_m=L,
                area_center_m2=A_pair,
                area_outer_m2=0.0,
                T_hot_K=STAGE_T[hot],
                T_cold_K=STAGE_T[cold],
                stage_hot=hot,
                stage_cold=cold,
            )
        )
    return DCBiasLine(segments=segments)


def total_stack_heat(
    n_coax: int = 16,
    n_dc: int = 8,
    drive_power_dBm: float = 0.0,
) -> Dict[str, float]:
    """Sum of heat loads per stage for an N-coax + M-DC wiring stack."""
    loads = {s: 0.0 for s in STAGES}
    coax = standard_coax_line(drive_power_dBm)
    dc = standard_dc_bias_line()
    for _ in range(n_coax):
        for s, q in coax.total_heat_by_stage().items():
            loads[s] += q
    for _ in range(n_dc):
        for s, q in dc.total_heat_by_stage().items():
            loads[s] += q
    return loads


def main():
    print("Per-line heat loads (BlueFors LD400-style geometry):")
    print()
    coax = standard_coax_line(drive_power_dBm=0.0)
    dc = standard_dc_bias_line()

    cond = coax.conducted_heat_by_stage()
    sig = coax.signal_dissipation_by_stage()
    print("Single coax line at +0 dBm drive:")
    print("  {:<12s} {:>14s} {:>14s} {:>14s}".format("stage", "conducted [W]", "signal [W]", "total [W]"))
    for s in STAGES:
        if s == "room":
            continue
        print("  {:<12s} {:>14.3e} {:>14.3e} {:>14.3e}".format(s, cond[s], sig[s], cond[s] + sig[s]))
    print()

    dc_loads = dc.conducted_heat_by_stage()
    print("Single DC bias line (phosphor-bronze, 36 AWG twisted pair):")
    print("  {:<12s} {:>14s}".format("stage", "conducted [W]"))
    for s in STAGES:
        if s == "room":
            continue
        print("  {:<12s} {:>14.3e}".format(s, dc_loads[s]))
    print()

    for n_coax, n_dc, dBm in [(16, 8, 0.0), (32, 16, 0.0), (64, 32, 0.0)]:
        loads = total_stack_heat(n_coax, n_dc, dBm)
        print("Stack: {} coax + {} DC at +{} dBm drive ->".format(n_coax, n_dc, dBm))
        for s in STAGES:
            if s == "room":
                continue
            print("  {:<12s} {:.3e} W".format(s, loads[s]))
        print()


if __name__ == "__main__":
    main()
