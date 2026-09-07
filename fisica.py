"""
Funciones físicas puras del Problema 2.1: Estadística de un semiconductor,
del nivel de Fermi a la recombinación.

Convenciones de unidades usadas en todo el módulo (deben respetarse al llamar
estas funciones desde app.py):
    - Energías: eV
    - Temperatura: K
    - Concentraciones (n, p, Nc, Nv, Na, Nd, delta_n): cm^-3
    - Masas efectivas: kg
    - Tiempos de vida: s
    - Movilidades: cm^2/(V*s)
    - Coeficientes de difusión: cm^2/s
    - Longitudes de difusión: cm
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from constantes import K_B_EV, K_B_J, Q, HBAR, H_PLANCK


# ---------------------------------------------------------------------------
# Estadística de Fermi-Dirac y densidad de estados
# ---------------------------------------------------------------------------

def fermi_dirac(E_eV, Ef_eV, T_K):
    """f(E): probabilidad de ocupación de un estado de energía E_eV, dado el
    nivel de Fermi Ef_eV y la temperatura T_K. Vectorizable con numpy."""
    x = (np.asarray(E_eV, dtype=float) - Ef_eV) / (K_B_EV * T_K)
    # forma numéricamente estable (evita overflow de exp para x grande)
    return np.where(x > 0, np.exp(-x) / (1.0 + np.exp(-x)), 1.0 / (1.0 + np.exp(x)))


def un_menos_fermi_dirac(E_eV, Ef_eV, T_K):
    """1-f(E): probabilidad de que el estado E_eV esté vacío (ocupado por un
    hueco). Se calcula como fermi_dirac(Ef_eV, E_eV, T_K) -intercambiando E y
    Ef- en vez de "1 - fermi_dirac(E_eV, Ef_eV, T_K)": cuando f(E)≈1 (estado
    muy por debajo de Ef), restar de 1 sufre cancelación catastrófica en
    float64 y pierde toda la información del valor real (verificado: error
    >30% en n·p=n_i² para dopajes altos sin este arreglo)."""
    return fermi_dirac(Ef_eV, E_eV, T_K)


def densidad_estados_conduccion(E_eV, Ec_eV, me_kg):
    """g_C(E) en cm^-3 eV^-1, para E >= Ec. g_C(E) = 0 para E < Ec.
    Fórmula: g_C(E) = (1/(2*pi^2)) * (2*me/hbar^2)^(3/2) * sqrt(E-Ec)  [SI],
    convertida de m^-3 J^-1 a cm^-3 eV^-1."""
    E_eV = np.asarray(E_eV, dtype=float)
    delta_E_J = np.clip(E_eV - Ec_eV, 0, None) * Q
    g_SI = (1.0 / (2 * np.pi**2)) * (2 * me_kg / HBAR**2)**1.5 * np.sqrt(delta_E_J)
    g_cm3_eV = g_SI * 1e-6 * Q  # m^-3 J^-1 -> cm^-3 eV^-1
    return np.where(E_eV >= Ec_eV, g_cm3_eV, 0.0)


def densidad_estados_valencia(E_eV, Ev_eV, mh_kg):
    """g_V(E) en cm^-3 eV^-1, para E <= Ev. g_V(E) = 0 para E > Ev."""
    E_eV = np.asarray(E_eV, dtype=float)
    delta_E_J = np.clip(Ev_eV - E_eV, 0, None) * Q
    g_SI = (1.0 / (2 * np.pi**2)) * (2 * mh_kg / HBAR**2)**1.5 * np.sqrt(delta_E_J)
    g_cm3_eV = g_SI * 1e-6 * Q
    return np.where(E_eV <= Ev_eV, g_cm3_eV, 0.0)


def Nc_efectiva(T_K, me_kg):
    """Densidad efectiva de estados en la banda de conducción, cm^-3."""
    Nc_SI = 2 * (2 * np.pi * me_kg * K_B_J * T_K / H_PLANCK**2)**1.5  # m^-3
    return Nc_SI * 1e-6


def Nv_efectiva(T_K, mh_kg):
    """Densidad efectiva de estados en la banda de valencia, cm^-3."""
    Nv_SI = 2 * (2 * np.pi * mh_kg * K_B_J * T_K / H_PLANCK**2)**1.5
    return Nv_SI * 1e-6


# ---------------------------------------------------------------------------
# n y p: vía aproximación de Boltzmann (cerrada) e integración numérica (TODO)
# ---------------------------------------------------------------------------

def n_boltzmann(Ef_eV, Ec_eV, T_K, Nc_cm3):
    """n = Nc * exp(-(Ec-Ef)/kT). Válida solo si Ec-Ef >> kT (no degenerado)."""
    return Nc_cm3 * np.exp(-(Ec_eV - Ef_eV) / (K_B_EV * T_K))


def p_boltzmann(Ef_eV, Ev_eV, T_K, Nv_cm3):
    """p = Nv * exp(-(Ef-Ev)/kT). Válida solo si Ef-Ev >> kT (no degenerado)."""
    return Nv_cm3 * np.exp(-(Ef_eV - Ev_eV) / (K_B_EV * T_K))


def n_numerico(Ef_eV, T_K, Ec_eV, me_kg, E_max_eV=None):
    """n = integral_{Ec}^{E_max} g_C(E)*f(E) dE, por cuadratura numérica.
    E_max_eV por defecto: Ec + 10*kT (la densidad de estados x Fermi decae
    exponencialmente más allá de eso).

    Se integra con el cambio de variable u = sqrt(E-Ec) (x = E-Ec = u^2), que
    elimina la singularidad de la derivada de sqrt(E-Ec) en el borde de banda
    (E=Ec). Sin este cambio, scipy.integrate.quad pierde precisión relativa
    por error de redondeo cuando Ef está lejos de Ec (caso muy no degenerado),
    como se verificó comparando contra la aproximación de Boltzmann (error
    numérico >30% sin el cambio de variable, <1% con él)."""
    if E_max_eV is None:
        # max(Ec, Ef) + 10kT: si el material está degenerado (Ef por dentro
        # de la banda), el límite debe extenderse más allá de Ef, no solo de Ec.
        E_max_eV = max(Ec_eV, Ef_eV) + 10.0 * K_B_EV * T_K
    u_max = np.sqrt(max(E_max_eV - Ec_eV, 0.0))
    integrando_u = lambda u: 2 * u * densidad_estados_conduccion(Ec_eV + u**2, Ec_eV, me_kg) \
        * fermi_dirac(Ec_eV + u**2, Ef_eV, T_K)
    valor, _ = quad(integrando_u, 0.0, u_max, limit=200)
    return valor


def p_numerico(Ef_eV, T_K, Ev_eV, mh_kg, E_min_eV=None):
    """p = integral_{E_min}^{Ev} g_V(E)*(1-f(E)) dE, análogo a n_numerico.
    Cambio de variable u = sqrt(Ev-E), mismo motivo que en n_numerico."""
    if E_min_eV is None:
        E_min_eV = min(Ev_eV, Ef_eV) - 10.0 * K_B_EV * T_K
    u_max = np.sqrt(max(Ev_eV - E_min_eV, 0.0))
    integrando_u = lambda u: 2 * u * densidad_estados_valencia(Ev_eV - u**2, Ev_eV, mh_kg) \
        * un_menos_fermi_dirac(Ev_eV - u**2, Ef_eV, T_K)
    valor, _ = quad(integrando_u, 0.0, u_max, limit=200)
    return valor


def resolver_ef(T_K, Na_cm3, Nd_cm3, Ec_eV, Ev_eV, me_kg, mh_kg):
    """Resuelve E_F numéricamente (scipy.optimize.brentq) imponiendo neutralidad
    de carga: p_numerico(Ef) + Nd - n_numerico(Ef) - Na = 0 (asumiendo
    ionización total de los dopantes; aproximación a declarar en la
    presentación). E_F NO se fija a mano: siempre sale de este solver.

    La función de carga neta es monótonamente decreciente en Ef (p baja y n
    sube al aumentar Ef), por lo que existe una única raíz en un intervalo
    suficientemente amplio alrededor de la banda prohibida."""

    def carga_neta(Ef_eV):
        n = n_numerico(Ef_eV, T_K, Ec_eV, me_kg)
        p = p_numerico(Ef_eV, T_K, Ev_eV, mh_kg)
        return p + Nd_cm3 - n - Na_cm3

    margen_eV = 1.0
    Ef_min = Ev_eV - margen_eV
    Ef_max = Ec_eV + margen_eV
    return brentq(carga_neta, Ef_min, Ef_max, xtol=1e-6, rtol=1e-10)


# ---------------------------------------------------------------------------
# Recombinación: los cuatro mecanismos y tiempo de vida efectivo
# ---------------------------------------------------------------------------

def tau_radiativo(n0_cm3, p0_cm3, delta_n_cm3, B_cm3_s):
    """tau_radiativa = 1 / [B*(n0+p0+delta_n)]."""
    return 1.0 / (B_cm3_s * (n0_cm3 + p0_cm3 + delta_n_cm3))


def tau_auger(delta_n_cm3, c_sum_cm6_s):
    """tau_Auger = 1 / [(cn+cp)*delta_n^2]."""
    return 1.0 / (c_sum_cm6_s * delta_n_cm3**2)


def tau_superficie(S_cm_s, W_cm):
    """Modelo simplificado (muestra delgada, perfil uniforme de exceso de
    portadores, recombinación en ambas caras): 1/tau_superficie = 2*S/W.
    Aproximación a declarar en la presentación."""
    return np.inf if S_cm_s <= 0 else W_cm / (2.0 * S_cm_s)


def tau_efectivo(tau_radiativo_s, tau_auger_s, tau_srh_s, tau_superficie_s):
    """1/tau_efectivo = suma de las 4 tasas inversas."""
    return 1.0 / (1.0 / tau_radiativo_s + 1.0 / tau_auger_s
                  + 1.0 / tau_srh_s + 1.0 / tau_superficie_s)


def coef_difusion_einstein(mu_cm2_Vs, T_K):
    """D = (kB*T/q) * mu, relación de Einstein. D en cm^2/s."""
    return (K_B_EV * T_K) * mu_cm2_Vs


def longitud_difusion(D_cm2_s, tau_s):
    """L = sqrt(D*tau), en cm."""
    return np.sqrt(D_cm2_s * tau_s)
