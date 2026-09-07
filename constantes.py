"""
Constantes físicas y parámetros de referencia del curso Celdas Solares (2026-2).
Cada constante lleva su unidad y su fuente (Anexo B del enunciado del Proyecto 1,
o la unidad correspondiente del curso: U2, U3, U4).

Ninguna constante usada en los cálculos debe declararse fuera de este módulo.
"""

# --- Constantes universales ---
K_B_EV = 8.6173e-5          # eV/K, constante de Boltzmann (Anexo B, U2)
K_B_J = 1.380649e-23        # J/K, constante de Boltzmann (SI)
Q = 1.602176634e-19         # C, carga elemental
H_PLANCK = 6.62607015e-34   # J·s, constante de Planck
HBAR = H_PLANCK / (2 * 3.141592653589793)  # J·s, hbar
M0 = 9.1093837015e-31       # kg, masa del electrón libre

# --- Parámetros de referencia del silicio a 300 K (Anexo B) ---
NI_SI_300K = 1.0e10         # cm^-3, concentración intrínseca del silicio a 300 K (U2)
EG_SI_300K = 1.12           # eV, ancho de banda prohibida del silicio a 300 K (U4)
KBT_Q_300K = 25.85e-3       # V, energía térmica kBT/q a 300 K (U4)

# --- Masas efectivas de densidad de estados del silicio (valores por defecto,
#     el usuario puede variarlas en la Pestaña 1) ---
ME_SI_DEFAULT = 1.08 * M0   # kg, masa efectiva de electrones (densidad de estados), Green (1990)
MH_SI_DEFAULT = 0.81 * M0   # kg, masa efectiva de huecos (densidad de estados), Green (1990)

# --- Movilidades de referencia (Anexo B, U3) usadas para el cálculo de
#     coeficientes de difusión D = (kB*T/q)*mu vía relación de Einstein ---
MU_N_REF = 1200.0           # cm^2/(V·s), electrones en base tipo p, NA~1e16 cm^-3, 300K (U3)
MU_P_REF = 60.0             # cm^2/(V·s), huecos en emisor tipo n+, ND~1e19-1e20 cm^-3, 300K (U3)

# --- Recombinación (U4, Anexo B) ---
B_RADIATIVO_SI = 4.73e-15   # cm^3/s, coeficiente radiativo del silicio (U4)
C_AUGER_SI = 4.0e-31        # cm^6/s, suma cn+cp, coeficiente Auger del silicio (U4)

# --- Semilla del grupo (Anexo A, Tabla A.1, fila S=7) ---
SEMILLA_S = 7
N_A_DEFAULT = 6.0e15        # cm^-3, concentración de aceptores (dopaje tipo p)
N_D_DEFAULT = 1.5e19        # cm^-3, concentración de donadores (dopaje tipo n), usada para explorar degeneración
TAU_SRH_VOLUMEN_DEFAULT = 300.0e-6   # s, tiempo de vida SRH de volumen asignado por semilla
S_FRONTAL_DEFAULT = 3.0e2   # cm/s, velocidad de recombinación superficial frontal asignada por semilla
T_OPERACION_DEFAULT = 25.0 + 273.15  # K, temperatura de operación asignada por semilla (25 °C)
ESPESOR_W_DEFAULT = 220.0e-4         # cm, espesor del bloque asignado por semilla (220 μm, Anexo A, S=7)
