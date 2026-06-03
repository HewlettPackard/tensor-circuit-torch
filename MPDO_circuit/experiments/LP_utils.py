"""
LP_utils.py
-----------
Lower-polariton (LP) dispersion and derived quantities.

All energies are in eV, lengths in µm, and times in ps unless stated
otherwise.

Constants
---------
Mc          : photonic mode energy (linear approximation), eV
Exca        : exciton energy, eV
OM          : half the Rabi splitting, eV
hbar_eV_ps  : ħ in eV·ps
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

Mc         = lambda k: 1.6114 + 0.037 * k   # photonic mode energy (eV), linear in k
Exca       = 1.6229                           # exciton energy (eV)
OM         = 3.4e-3                           # half Rabi splitting (eV)
hbar_eV_ps = 6.582e-4                         # ħ in eV·ps


# ---------------------------------------------------------------------------
# Dispersion and derived quantities
# ---------------------------------------------------------------------------

def hbar_omega_LP(k: float) -> float:
    """
    Lower-polariton energy (eV) at in-plane wavevector k (µm⁻¹).

    Uses the standard two-mode coupling formula:
        E_LP = ½(E_c + E_x) - ½ √((E_c - E_x)² + 4Ω²)

    Parameters
    ----------
    k : in-plane wavevector (µm⁻¹).

    Returns
    -------
    float or np.ndarray
        LP energy in eV.
    """
    return 0.5 * (Mc(k) + Exca) - 0.5 * np.sqrt((Mc(k) - Exca) ** 2 + 4.0 * OM ** 2)


def vg_LP(k: float, dk: float = 1e-3) -> np.ndarray:
    """
    Lower-polariton group velocity (µm/ps) at wavevector k (µm⁻¹).

    Computed via a central finite difference of the dispersion:
        v_g = (1/ħ) dE/dk

    Parameters
    ----------
    k  : in-plane wavevector (µm⁻¹).
    dk : finite-difference step size (µm⁻¹); default 1e-3.

    Returns
    -------
    float or np.ndarray
        Group velocity in µm/ps.
    """
    return (hbar_omega_LP(k + dk / 2.0) - hbar_omega_LP(k - dk / 2.0)) / dk / hbar_eV_ps


def curvature_LP(k: float, dk: float = 1e-3) -> np.ndarray:
    """
    Lower-polariton band curvature d²E/dk² (µm²/ps) at wavevector k (µm⁻¹).

    Proportional to the inverse effective mass.  Computed as the finite-
    difference derivative of the group velocity:
        d²E/dk² = dv_g/dk

    Parameters
    ----------
    k  : in-plane wavevector (µm⁻¹).
    dk : finite-difference step size (µm⁻¹); default 1e-3.

    Returns
    -------
    float or np.ndarray
        Band curvature in µm²/ps.
    """
    return (vg_LP(k + dk / 2.0, dk) - vg_LP(k - dk / 2.0, dk)) / dk


def exciton_fraction(k: float) -> float:
    """
    Excitonic Hopfield coefficient |u_k|² at wavevector k (µm⁻¹).

    Gives the exciton weight of the lower-polariton mode.  Used to rescale
    the bare exciton–exciton interaction to the effective polariton
    nonlinearity via g_eff = |u_k|⁴ · g_ex.

    Parameters
    ----------
    k : in-plane wavevector (µm⁻¹).

    Returns
    -------
    float or np.ndarray
        Dimensionless exciton fraction in [0, 1].
    """
    theta_k = np.arctan(OM / (hbar_omega_LP(k) - Mc(k)))
    return np.cos(theta_k) ** 2