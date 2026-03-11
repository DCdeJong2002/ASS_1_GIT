"""
Blade Element Momentum (BEM) model for the wind-turbine case.

This script consolidates the BEM notebook implementation into a single Python file.
It is based on the uploaded notebook and keeps the same core model structure:
- axial and tangential induction
- Glauert correction for heavily loaded rotors
- Prandtl tip/root loss correction
- blade-element force evaluation from airfoil polars
- streamtube-by-streamtube BEM solution

Expected polar file format
--------------------------
A whitespace-separated text/CSV file with four columns:
    alpha  cl  cd  cm
Only alpha, cl, and cd are used.

Example usage
-------------
1. Put your polar file (for example DU95W180.cvs) in the same folder.
2. Run:
       python bem_wind_turbine.py
3. The script prints CT and CP and shows the standard radial-distribution plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Core momentum relations
# ============================================================

def CTfunction(a: np.ndarray | float, glauert: bool = False) -> np.ndarray:
    """
    Compute thrust coefficient C_T as a function of axial induction factor a.

    Parameters
    ----------
    a : array-like or float
        Axial induction factor.
    glauert : bool, default=False
        If True, apply Glauert's high-loading correction to the momentum curve.

    Returns
    -------
    np.ndarray
        Thrust coefficient corresponding to the supplied induction factor.
    """
    a = np.asarray(a, dtype=float)
    CT = 4.0 * a * (1.0 - a)

    if glauert:
        CT1 = 1.816
        a1 = 1.0 - np.sqrt(CT1) / 2.0
        mask = a > a1
        CT[mask] = CT1 - 4.0 * (np.sqrt(CT1) - 1.0) * (1.0 - a[mask])

    return CT


def ainduction(CT: np.ndarray | float) -> np.ndarray:
    """
    Compute axial induction factor a from thrust coefficient C_T.

    Glauert's correction is included for heavily loaded conditions.

    Parameters
    ----------
    CT : array-like or float
        Thrust coefficient.

    Returns
    -------
    np.ndarray
        Axial induction factor.
    """
    CT = np.asarray(CT, dtype=float)
    a = np.zeros(np.shape(CT), dtype=float)

    CT1 = 1.816
    CT2 = 2.0 * np.sqrt(CT1) - CT1

    mask_high = CT >= CT2
    mask_low = ~mask_high

    a[mask_high] = 1.0 + (CT[mask_high] - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
    a[mask_low] = 0.5 - 0.5 * np.sqrt(1.0 - CT[mask_low])

    return a


def PrandtlTipRootCorrection(
    r_R: np.ndarray | float,
    rootradius_R: float,
    tipradius_R: float,
    TSR: float,
    NBlades: int,
    axial_induction: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the combined Prandtl tip and root loss correction.

    Parameters
    ----------
    r_R : array-like or float
        Local radius nondimensionalized by rotor radius.
    rootradius_R : float
        Blade root location r_root / R.
    tipradius_R : float
        Blade tip location r_tip / R.
    TSR : float
        Tip-speed ratio, Omega * R / U_inf.
    NBlades : int
        Number of blades.
    axial_induction : array-like or float
        Local axial induction factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Combined Prandtl factor, tip factor, root factor.
    """
    r_R = np.asarray(r_R, dtype=float)
    axial_induction = np.asarray(axial_induction, dtype=float)

    sqrt_term = np.sqrt(1.0 + ((TSR * r_R) ** 2) / ((1.0 - axial_induction) ** 2))

    temp_tip = -NBlades / 2.0 * (tipradius_R - r_R) / r_R * sqrt_term
    Ftip = np.array(2.0 / np.pi * np.arccos(np.exp(temp_tip)))
    Ftip[np.isnan(Ftip)] = 0.0

    temp_root = NBlades / 2.0 * (rootradius_R - r_R) / r_R * sqrt_term
    Froot = np.array(2.0 / np.pi * np.arccos(np.exp(temp_root)))
    Froot[np.isnan(Froot)] = 0.0

    return Froot * Ftip, Ftip, Froot


# ============================================================
# Airfoil polar handling
# ============================================================

def load_airfoil_polar(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load airfoil polar data from a whitespace-separated file.

    The file is expected to contain four columns:
    alpha, cl, cd, cm.

    Parameters
    ----------
    filepath : str or Path
        Path to the polar file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of alpha [deg], cl, and cd.
    """
    filepath = Path(filepath)
    data = pd.read_csv(
        filepath,
        header=0,
        names=["alpha", "cl", "cd", "cm"],
        sep=r"\s+",
        engine="python",
    )

    polar_alpha = data["alpha"].to_numpy(dtype=float)
    polar_cl = data["cl"].to_numpy(dtype=float)
    polar_cd = data["cd"].to_numpy(dtype=float)

    return polar_alpha, polar_cl, polar_cd


# ============================================================
# Blade element aerodynamics
# ============================================================

def loadBladeElement(
    vnorm: float,
    vtan: float,
    r_R: float,
    chord: float,
    twist: float,
    polar_alpha: np.ndarray,
    polar_cl: np.ndarray,
    polar_cd: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute 2D blade-element loads.

    Parameters
    ----------
    vnorm : float
        Axial velocity component at the rotor plane.
    vtan : float
        Tangential velocity component at the rotor plane.
    r_R : float
        Nondimensional radial position. Included for interface consistency.
    chord : float
        Local chord [m].
    twist : float
        Local geometric twist angle [deg].
    polar_alpha, polar_cl, polar_cd : np.ndarray
        Airfoil polar arrays used for interpolation.

    Returns
    -------
    tuple[float, float, float]
        Normal force per unit span, tangential force per unit span, circulation.
    """
    _ = r_R  # kept for compatibility with the notebook formulation

    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm, vtan)
    alpha = twist + inflowangle * 180.0 / np.pi

    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)

    lift = 0.5 * vmag2 * cl * chord
    drag = 0.5 * vmag2 * cd * chord

    fnorm = lift * np.cos(inflowangle) + drag * np.sin(inflowangle)
    ftan = lift * np.sin(inflowangle) - drag * np.cos(inflowangle)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord

    return float(fnorm), float(ftan), float(gamma)


def solveStreamtube(
    Uinf: float,
    r1_R: float,
    r2_R: float,
    rootradius_R: float,
    tipradius_R: float,
    Omega: float,
    Radius: float,
    NBlades: int,
    chord: float,
    twist: float,
    polar_alpha: np.ndarray,
    polar_cl: np.ndarray,
    polar_cd: np.ndarray,
) -> list[float]:
    """
    Solve one annular streamtube by balancing blade-element and momentum loads.

    Parameters
    ----------
    Uinf : float
        Freestream wind speed [m/s].
    r1_R, r2_R : float
        Inner and outer annulus radius nondimensionalized by R.
    rootradius_R, tipradius_R : float
        Root and tip positions nondimensionalized by R.
    Omega : float
        Rotational speed [rad/s].
    Radius : float
        Rotor radius [m].
    NBlades : int
        Number of blades.
    chord : float
        Local chord [m].
    twist : float
        Local twist [deg].
    polar_alpha, polar_cl, polar_cd : np.ndarray
        Airfoil polar arrays.

    Returns
    -------
    list[float]
        [a, a_prime, r_mid/R, fnorm, ftan, gamma]
    """
    Area = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_R = 0.5 * (r1_R + r2_R)

    a = 0.0
    aline = 0.0

    Niterations = 100
    Erroriterations = 1e-5

    for _ in range(Niterations):
        # Local velocities at the rotor plane
        Urotor = Uinf * (1.0 - a)
        Utan = (1.0 + aline) * Omega * r_R * Radius

        # 2D sectional loads
        fnorm, ftan, gamma = loadBladeElement(
            Urotor, Utan, r_R, chord, twist, polar_alpha, polar_cl, polar_cd
        )

        # Convert sectional load to annulus load
        load3Daxial = fnorm * Radius * (r2_R - r1_R) * NBlades

        # Streamtube thrust coefficient
        CT = load3Daxial / (0.5 * Area * Uinf**2)

        # Updated axial induction with Glauert correction
        anew = ainduction(CT)

        # Prandtl correction
        TSR = Omega * Radius / Uinf
        Prandtl, _, _ = PrandtlTipRootCorrection(
            r_R, rootradius_R, tipradius_R, TSR, NBlades, anew
        )
        if Prandtl < 1e-4:
            Prandtl = 1e-4

        anew = anew / Prandtl

        # Relaxation for better convergence
        a = 0.75 * a + 0.25 * anew

        # Tangential induction
        aline = ftan * NBlades / (
            2.0 * np.pi * Uinf * (1.0 - a) * Omega * 2.0 * (r_R * Radius) ** 2
        )
        aline = aline / Prandtl

        # Convergence check
        if np.abs(a - anew) < Erroriterations:
            break

    return [float(a), float(aline), float(r_R), float(fnorm), float(ftan), float(gamma)]


# ============================================================
# Rotor-level assembly
# ============================================================

def run_bem_case(
    polar_file: str | Path,
    Uinf: float = 10.0,
    TSR: float = 8.0,
    Radius: float = 50.0,
    NBlades: int = 3,
    pitch: float = 2.0,
    root_location_R: float = 0.2,
    tip_location_R: float = 1.0,
    delta_r_R: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run the full BEM computation for the default wind-turbine case.

    The geometry is the same as in the notebook:
    - chord(r/R) = 3 * (1 - r/R) + 1
    - twist(r/R) = -14 * (1 - r/R) + pitch

    Parameters
    ----------
    polar_file : str or Path
        Path to the airfoil polar file.
    Uinf : float, default=10.0
        Freestream wind speed [m/s].
    TSR : float, default=8.0
        Tip-speed ratio.
    Radius : float, default=50.0
        Rotor radius [m].
    NBlades : int, default=3
        Number of blades.
    pitch : float, default=2.0
        Collective pitch [deg].
    root_location_R : float, default=0.2
        Root position nondimensionalized by radius.
    tip_location_R : float, default=1.0
        Tip position nondimensionalized by radius.
    delta_r_R : float, default=0.01
        Radial annulus width in nondimensional form.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict]
        r-grid, results array, radial chord distribution, and summary metrics.
    """
    polar_alpha, polar_cl, polar_cd = load_airfoil_polar(polar_file)

    r_R = np.arange(root_location_R, tip_location_R + delta_r_R / 2.0, delta_r_R)

    chord_distribution = 3.0 * (1.0 - r_R) + 1.0
    twist_distribution = -14.0 * (1.0 - r_R) + pitch

    Omega = Uinf * TSR / Radius

    results = np.zeros((len(r_R) - 1, 6), dtype=float)

    for i in range(len(r_R) - 1):
        r_mid = 0.5 * (r_R[i] + r_R[i + 1])
        chord = np.interp(r_mid, r_R, chord_distribution)
        twist = np.interp(r_mid, r_R, twist_distribution)

        results[i, :] = solveStreamtube(
            Uinf,
            r_R[i],
            r_R[i + 1],
            root_location_R,
            tip_location_R,
            Omega,
            Radius,
            NBlades,
            chord,
            twist,
            polar_alpha,
            polar_cl,
            polar_cd,
        )

    dr = (r_R[1:] - r_R[:-1]) * Radius
    CT = np.sum(dr * results[:, 3] * NBlades / (0.5 * Uinf**2 * np.pi * Radius**2))
    CP = np.sum(
        dr
        * results[:, 4]
        * results[:, 2]
        * NBlades
        * Radius
        * Omega
        / (0.5 * Uinf**3 * np.pi * Radius**2)
    )

    summary = {
        "Uinf_m_per_s": Uinf,
        "TSR": TSR,
        "Radius_m": Radius,
        "Omega_rad_per_s": Omega,
        "NBlades": NBlades,
        "CT": float(CT),
        "CP": float(CP),
    }

    return r_R, results, chord_distribution, summary


# ============================================================
# Plotting helpers
# ============================================================

def plot_results(Uinf: float, Radius: float, Omega: float, NBlades: int, results: np.ndarray) -> None:
    """Generate the same result plots as in the notebook."""

    plt.figure(figsize=(12, 6))
    plt.title("Axial and tangential induction")
    plt.plot(results[:, 2], results[:, 0], "r-", label=r"$a$")
    plt.plot(results[:, 2], results[:, 1], "g--", label=r"$a'$")
    plt.grid(True)
    plt.xlabel("r/R")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.title(r"Normal and tangential force, nondimensionalized by $\frac{1}{2}\rho U_\infty^2 R$")
    plt.plot(results[:, 2], results[:, 3] / (0.5 * Uinf**2 * Radius), "r-", label="Fnorm")
    plt.plot(results[:, 2], results[:, 4] / (0.5 * Uinf**2 * Radius), "g--", label="Ftan")
    plt.grid(True)
    plt.xlabel("r/R")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.title(r"Circulation distribution, nondimensionalized by $\frac{\pi U_\infty^2}{\Omega N_{blades}}$")
    plt.plot(results[:, 2], results[:, 5] / (np.pi * Uinf**2 / (NBlades * Omega)), "r-", label=r"$\Gamma$")
    plt.grid(True)
    plt.xlabel("r/R")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Main execution block
# ============================================================

def main() -> None:
    """Run the default wind-turbine BEM example."""
    polar_file = Path("DU95W180.cvs")

    if not polar_file.exists():
        raise FileNotFoundError(
            "Polar file not found. Place the airfoil polar file (for example 'DU95W180.cvs') "
            "next to this script or edit the 'polar_file' path in main()."
        )

    Uinf = 10.0
    Radius = 50.0
    TSR = 10.0
    NBlades = 3

    _, results, _, summary = run_bem_case(
        polar_file=polar_file,
        Uinf=Uinf,
        TSR=TSR,
        Radius=Radius,
        NBlades=NBlades,
        pitch=2.0,
        root_location_R=0.2,
        tip_location_R=1.0,
        delta_r_R=0.01,
    )

    print(f"CT = {summary['CT']:.6f}")
    print(f"CP = {summary['CP']:.6f}")

    plot_results(
        Uinf=Uinf,
        Radius=Radius,
        Omega=summary["Omega_rad_per_s"],
        NBlades=NBlades,
        results=results,
    )


if __name__ == "__main__":
    main()