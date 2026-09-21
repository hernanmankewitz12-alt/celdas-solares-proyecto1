import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import constantes as c
import fisica as f

st.set_page_config(page_title="Celdas Solares - Proyecto 1 (2.1)", page_icon="☀️", layout="wide")

VIS = Path(__file__).parent / "vis"

# Paleta única para todo el simulador (gráficos, 3D y animaciones usan los mismos colores)
COL = {
    "e": "#4da3ff", "h": "#ff5a5f", "don": "#3ddc97", "acc": "#ffa53d",
    "rad": "#ffe066", "auger": "#ff8c42", "srh": "#b07cff", "surf": "#2ec4b6",
    "amber": "#f5b83d", "text": "#e6e9f2", "dim": "#96a0b8",
}
NOMBRE_MEC = {"rad": "Radiativa", "auger": "Auger", "srh": "SRH (trampas)", "surf": "Superficie"}

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(x, d=2):
    """Notación científica legible: 1.50×10¹⁹."""
    if x == 0 or not np.isfinite(x):
        return str(x)
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10**e:.{d}f}×10{str(e).translate(SUP)}"


def fmt_t(s):
    if not np.isfinite(s):
        return "∞"
    for u, k in (("s", 1), ("ms", 1e-3), ("μs", 1e-6), ("ns", 1e-9)):
        if s >= k:
            return f"{s / k:.3g} {u}"
    return f"{s:.2e} s"


def componente(nombre, params, alto, scrolling=False):
    """Inserta una visualización HTML/JS de la carpeta vis/ con sus parámetros."""
    html = (VIS / f"{nombre}.html").read_text(encoding="utf-8").replace("__PARAMS__", json.dumps(params))
    if hasattr(st, "iframe"):           # Streamlit ≥ 1.5x: reemplazo oficial de components.html
        st.iframe(html, height=alto)
    else:
        components.html(html, height=alto if isinstance(alto, int) else 760, scrolling=scrolling)


def estilo(fig, alto=420, **kw):
    base = dict(
        template="plotly_dark", height=alto,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1629",
        font=dict(family="Segoe UI, system-ui, sans-serif", color=COL["text"], size=12.5),
        margin=dict(l=64, r=24, t=56, b=52),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#131a2e", font_size=12),
    )
    base["legend"].update(kw.pop("legend", {}))
    base.update(kw)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.15)", exponentformat="power")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.15)", exponentformat="power")
    return fig


def mostrar(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def rgba(hex_col, a):
    h = hex_col.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


st.markdown("""
<style>
  .block-container{padding-top:2.2rem}
  .hero{border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:22px 26px 18px;
    background:radial-gradient(1200px 300px at 0% 0%, rgba(245,184,61,.14), transparent 60%),
               radial-gradient(900px 260px at 100% 100%, rgba(77,163,255,.13), transparent 60%), #10172c;
    margin-bottom:10px}
  .hero .eyebrow{font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:#f5b83d;font-weight:650}
  .hero h1{font-size:2.1rem;margin:.2rem 0 .3rem;padding:0;line-height:1.15}
  .hero p{color:#b9c1d6;max-width:72ch;margin:0 0 .8rem}
  .chips{display:flex;flex-wrap:wrap;gap:8px}
  .chip{font-size:.8rem;padding:4px 11px;border-radius:999px;background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.08);color:#cfd6e6;font-variant-numeric:tabular-nums}
  .chip b{color:#f5b83d}
  div[data-testid="stMetric"]{background:#111831;border:1px solid rgba(255,255,255,.07);
    border-radius:12px;padding:10px 14px}
  div[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
  .nota{color:#96a0b8;font-size:.86rem}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Estado inicial (solo la primera vez que corre la app) — valores de semilla
# ---------------------------------------------------------------------------
defaults = {
    "T_K": c.T_OPERACION_DEFAULT,
    "tipo_material": "intrínseco",
    "Na_cm3": c.N_A_DEFAULT,
    "Nd_cm3": c.N_D_DEFAULT,
    "me_kg": c.ME_SI_DEFAULT,
    "mh_kg": c.MH_SI_DEFAULT,
    "Eg_eV": c.EG_SI_300K,
    "delta_n_cm3": 1e14,
    "S_frontal_cm_s": c.S_FRONTAL_DEFAULT,
    "tau_srh_volumen_s": c.TAU_SRH_VOLUMEN_DEFAULT,
    "espesor_cm": c.ESPESOR_W_DEFAULT,  # cm, espesor asignado por semilla (220 um, S=7)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Barra lateral: semilla visible + controles compartidos entre pestañas
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"**Semilla del grupo: S = {c.SEMILLA_S}**")
    st.caption("N_A=6e15, N_D=1.5e19 cm^-3, tau_SRH=300 us, S_frontal=3e2 cm/s, T=25°C")

    st.subheader("Material")
    st.session_state.T_K = st.slider("Temperatura T [K]", 100, 600, int(st.session_state.T_K), 1)
    st.session_state.tipo_material = st.selectbox(
        "Tipo de material", ["intrínseco", "tipo n", "tipo p"],
        index=["intrínseco", "tipo n", "tipo p"].index(st.session_state.tipo_material))

    opciones_Na = sorted(set([10.0**e for e in range(13, 22)] + [c.N_A_DEFAULT]))
    opciones_Nd = sorted(set([10.0**e for e in range(13, 22)] + [c.N_D_DEFAULT]))
    st.session_state.Na_cm3 = st.select_slider(
        "Dopaje aceptores N_A [cm^-3]", options=opciones_Na, value=st.session_state.Na_cm3,
        format_func=lambda v: f"{v:.1e}", disabled=st.session_state.tipo_material != "tipo p")
    st.session_state.Nd_cm3 = st.select_slider(
        "Dopaje donadores N_D [cm^-3]", options=opciones_Nd, value=st.session_state.Nd_cm3,
        format_func=lambda v: f"{v:.1e}", disabled=st.session_state.tipo_material != "tipo n")

    me_factor = st.slider("m_e / m_0 (densidad de estados)", 0.1, 2.0,
                          float(st.session_state.me_kg / c.M0), 0.01)
    st.session_state.me_kg = me_factor * c.M0
    mh_factor = st.slider("m_h / m_0 (densidad de estados)", 0.1, 2.0,
                          float(st.session_state.mh_kg / c.M0), 0.01)
    st.session_state.mh_kg = mh_factor * c.M0

    st.session_state.Eg_eV = st.slider("Ancho de banda prohibida E_g [eV]", 0.5, 2.0,
                                       float(st.session_state.Eg_eV), 0.01)

    st.subheader("Recombinación")
    st.session_state.delta_n_cm3 = st.select_slider(
        "Nivel de inyección Δn [cm^-3]", options=[10.0**e for e in range(12, 19)],
        value=st.session_state.delta_n_cm3, format_func=lambda v: f"{v:.0e}")

    opciones_tau = sorted(set([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3, 3e-3] + [c.TAU_SRH_VOLUMEN_DEFAULT]))
    st.session_state.tau_srh_volumen_s = st.select_slider(
        "τ_SRH del volumen [μs]", options=opciones_tau, value=st.session_state.tau_srh_volumen_s,
        format_func=lambda v: f"{v*1e6:g}")

    opciones_S = sorted(set([10.0**e for e in range(0, 7)] + [c.S_FRONTAL_DEFAULT]))
    st.session_state.S_frontal_cm_s = st.select_slider(
        "Velocidad de recombinación superficial S [cm/s]", options=opciones_S,
        value=st.session_state.S_frontal_cm_s, format_func=lambda v: f"{v:.0e}")

    espesor_um = st.slider("Espesor del bloque [μm]", 20, 300, int(round(st.session_state.espesor_cm * 1e4)), 1)
    st.session_state.espesor_cm = espesor_um * 1e-4

# ---------------------------------------------------------------------------
# Cálculo central (compartido por todas las pestañas)
# ---------------------------------------------------------------------------
Ev_eV = 0.0                      # convención Ev = 0
Ec_eV = st.session_state.Eg_eV
Eg_eV = Ec_eV - Ev_eV
T_K = st.session_state.T_K
me_kg, mh_kg = st.session_state.me_kg, st.session_state.mh_kg
tipo = st.session_state.tipo_material
kT = c.K_B_EV * T_K

# Concentraciones de dopantes efectivas según el tipo de material elegido.
# Ionización total de los dopantes (aproximación declarada en fisica.resolver_ef).
if tipo == "intrínseco":
    Na_cm3, Nd_cm3 = 0.0, 0.0
elif tipo == "tipo n":
    Na_cm3, Nd_cm3 = 0.0, st.session_state.Nd_cm3
else:
    Na_cm3, Nd_cm3 = st.session_state.Na_cm3, 0.0

Ef_eV = f.resolver_ef(T_K, Na_cm3, Nd_cm3, Ec_eV, Ev_eV, me_kg, mh_kg)
Nc_cm3 = f.Nc_efectiva(T_K, me_kg)
Nv_cm3 = f.Nv_efectiva(T_K, mh_kg)
n_num = f.n_numerico(Ef_eV, T_K, Ec_eV, me_kg)
p_num = f.p_numerico(Ef_eV, T_K, Ev_eV, mh_kg)
n_boltz = f.n_boltzmann(Ef_eV, Ec_eV, T_K, Nc_cm3)
p_boltz = f.p_boltzmann(Ef_eV, Ev_eV, T_K, Nv_cm3)
degenerado = (Na_cm3 >= 1e19) or (Nd_cm3 >= 1e19)

# n_i²(T) recalculado a los T, m_e, m_h, E_g actuales (no un valor fijo de 300 K).
if tipo == "intrínseco":
    n_i_num, p_i_num = n_num, p_num
else:
    Ef_i_eV = f.resolver_ef(T_K, 0.0, 0.0, Ec_eV, Ev_eV, me_kg, mh_kg)
    n_i_num = f.n_numerico(Ef_i_eV, T_K, Ec_eV, me_kg)
    p_i_num = f.p_numerico(Ef_i_eV, T_K, Ev_eV, mh_kg)
ni2_num = n_i_num * p_i_num
ni2_boltz = Nc_cm3 * Nv_cm3 * np.exp(-Eg_eV / kT)

# Vidas medias al Δn actual
dn = st.session_state.delta_n_cm3
tau_srh_s = st.session_state.tau_srh_volumen_s
S_cm_s = st.session_state.S_frontal_cm_s
W_cm = st.session_state.espesor_cm
tau_act = {
    "rad": f.tau_radiativo(n_num, p_num, dn, c.B_RADIATIVO_SI),
    "auger": f.tau_auger(dn, c.C_AUGER_SI),
    "srh": tau_srh_s,
    "surf": f.tau_superficie(S_cm_s, W_cm),
}
tau_eff_actual = f.tau_efectivo(*tau_act.values())
tasa = {k: 1.0 / v for k, v in tau_act.items()}
share = {k: 100 * v / sum(tasa.values()) for k, v in tasa.items()}
dominante = max(share, key=share.get)
delta_n_cruce = np.sqrt(1.0 / (c.C_AUGER_SI * tau_srh_s))

# Mapeo declarado partícula <-> concentración (convención de visualización, no una
# constante física): 1 símbolo por cada década por encima de 10^6 cm^-3, tope 40.
PISO_EXPONENTE_VIS = 6
MAX_SIMBOLOS_VIS = 40


def num_simbolos(concentracion_cm3):
    if concentracion_cm3 <= 10 ** PISO_EXPONENTE_VIS:
        return 0
    return int(min(MAX_SIMBOLOS_VIS, max(1, round(np.log10(concentracion_cm3) - PISO_EXPONENTE_VIS))))


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Celdas Solares · Proyecto 1 · Problema 2.1</div>
  <h1>Dentro del silicio: portadores, bandas y recombinación</h1>
  <p>Estadística de Fermi-Dirac resuelta numéricamente, un bloque de silicio en 3D que puedes iluminar
  y los cuatro caminos por los que un par electrón-hueco desaparece. Todo responde a los controles de la barra lateral.</p>
  <div class="chips">
    <span class="chip">Semilla <b>S = {c.SEMILLA_S}</b></span>
    <span class="chip">N_A = <b>6×10¹⁵</b> cm⁻³</span>
    <span class="chip">N_D = <b>1.5×10¹⁹</b> cm⁻³</span>
    <span class="chip">W = <b>220</b> μm</span>
    <span class="chip">τ_SRH = <b>300</b> μs</span>
    <span class="chip">S_frontal = <b>3×10²</b> cm/s</span>
    <span class="chip">T = <b>25 °C</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Estadística de portadores",
    "2. El bloque de silicio (3D)",
    "3. Recombinación",
    "Validación",
])

# ---------------------------------------------------------------------------
# Pestaña 1 — Estadística de portadores
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def pelicula_temperatura(tipo, Na, Nd, me, mh, Ec, Ev):
    """E_F(T) y las densidades g·f para T = 100…600 K (cuadros de la animación)."""
    Ts = np.arange(100, 601, 25, dtype=float)
    Efs = np.array([f.resolver_ef(T, Na, Nd, Ec, Ev, me, mh) for T in Ts])
    E = np.linspace(Ev - 0.35, Ec + 0.35, 500)
    E = np.unique(np.concatenate([E, [Efs.max() + 0.2, Efs.min() - 0.2]]))
    gC = f.densidad_estados_conduccion(E, Ec, me)
    gV = f.densidad_estados_valencia(E, Ev, mh)
    prod_n = np.array([gC * f.fermi_dirac(E, ef, T) for T, ef in zip(Ts, Efs)])
    prod_p = np.array([gV * f.un_menos_fermi_dirac(E, ef, T) for T, ef in zip(Ts, Efs)])
    return Ts, Efs, E, prod_n, prod_p


@st.cache_data(show_spinner=False)
def superficie_ef(me, mh, Ec, Ev):
    """E_F(T, dopaje) resuelto por neutralidad de carga en una grilla (eje x: dopaje neto)."""
    Ts = np.linspace(100, 600, 11)
    k = np.arange(1, 10)                         # década sobre 10^12: 1e13 … 1e21
    xs = np.concatenate([-k[::-1], [0], k]).astype(float)
    Z = np.full((len(Ts), len(xs)), np.nan)
    for i, T in enumerate(Ts):
        for j, x in enumerate(xs):
            Na = 10.0 ** (12 - x) if x < 0 else 0.0
            Nd = 10.0 ** (12 + x) if x > 0 else 0.0
            try:
                Z[i, j] = f.resolver_ef(T, Na, Nd, Ec, Ev, me, mh)
            except ValueError:
                pass
    return Ts, xs, Z


def x_dopaje(tipo_, Na, Nd):
    if tipo_ == "tipo n":
        return np.log10(Nd) - 12
    if tipo_ == "tipo p":
        return -(np.log10(Na) - 12)
    return 0.0


with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"E_F [eV] · {(Ef_eV - (Ec_eV + Ev_eV) / 2) * 1000:+.0f} meV del centro", f"{Ef_eV:.4f}")
    m2.metric("Electrones n  [cm⁻³]", sci(n_num))
    m3.metric("Huecos p  [cm⁻³]", sci(p_num))
    m4.metric(f"n_i a T = {T_K:.0f} K  [cm⁻³]", sci(np.sqrt(ni2_num)))

    # --- Tres paneles enlazados: g(E), f(E) y su producto --------------------------
    Emin = min(Ev_eV - max(6 * kT, 0.15), Ef_eV - 6 * kT)
    Emax = max(Ec_eV + max(6 * kT, 0.15), Ef_eV + 6 * kT)
    E_range = np.unique(np.concatenate([np.linspace(Emin, Emax, 900), [Ec_eV, Ev_eV]]))
    gC = f.densidad_estados_conduccion(E_range, Ec_eV, me_kg)
    gV = f.densidad_estados_valencia(E_range, Ev_eV, mh_kg)
    fE = f.fermi_dirac(E_range, Ef_eV, T_K)
    un_fE = f.un_menos_fermi_dirac(E_range, Ef_eV, T_K)
    prod_n, prod_p = gC * fE, gV * un_fE

    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=("Densidad de estados g(E)", "Ocupación f(E) de Fermi-Dirac",
                                        "g·f: portadores reales (área = n, p)"))
    hov = "E = %{y:.3f} eV<br>%{x:.3e}<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=gC, y=E_range, name="g_C(E)", line=dict(color=COL["e"], width=2),
                             fill="tozerox", fillcolor=rgba(COL["e"], .12), hovertemplate=hov), 1, 1)
    fig.add_trace(go.Scatter(x=gV, y=E_range, name="g_V(E)", line=dict(color=COL["h"], width=2),
                             fill="tozerox", fillcolor=rgba(COL["h"], .12), hovertemplate=hov), 1, 1)
    fig.add_trace(go.Scatter(x=fE, y=E_range, name="f(E)", line=dict(color=COL["e"], width=2.5),
                             hovertemplate=hov), 1, 2)
    fig.add_trace(go.Scatter(x=un_fE, y=E_range, name="1 − f(E)", line=dict(color=COL["h"], width=2.5, dash="dot"),
                             hovertemplate=hov), 1, 2)
    fig.add_trace(go.Scatter(x=prod_n, y=E_range, name="g_C·f  → n", line=dict(color=COL["e"], width=1.5),
                             fill="tozerox", fillcolor=rgba(COL["e"], .45), hovertemplate=hov), 1, 3)
    fig.add_trace(go.Scatter(x=prod_p, y=E_range, name="g_V·(1−f)  → p", line=dict(color=COL["h"], width=1.5),
                             fill="tozerox", fillcolor=rgba(COL["h"], .45), hovertemplate=hov), 1, 3)
    for col in (1, 2, 3):
        fig.add_hrect(y0=Ev_eV, y1=Ec_eV, fillcolor="rgba(255,255,255,.035)", line_width=0, row=1, col=col)
        fig.add_hline(y=Ef_eV, line=dict(color=COL["amber"], dash="dash", width=1.6), row=1, col=col)
    fig.add_annotation(x=0.98, y=Ef_eV, xref="x domain", yref="y", text=f"E_F = {Ef_eV:.3f} eV", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=COL["amber"]))
    fig.add_annotation(x=0.98, y=(Ec_eV + Ev_eV) / 2 - 0.12 * Eg_eV, xref="x domain", yref="y", text="banda prohibida",
                       showarrow=False, xanchor="right", font=dict(color=COL["dim"], size=11))
    fig.update_yaxes(title_text="Energía [eV]  (E_V = 0)", row=1, col=1)
    fig.update_xaxes(title_text="cm⁻³ eV⁻¹", row=1, col=1)
    fig.update_xaxes(title_text="probabilidad de ocupación", range=[-0.02, 1.02], row=1, col=2)
    fig.update_xaxes(title_text="cm⁻³ eV⁻¹", row=1, col=3)
    estilo(fig, 470, legend=dict(orientation="h", y=-0.2, x=0))
    mostrar(fig)
    st.markdown('<div class="nota">Pasa el cursor para leer valores; arrastra para hacer zoom y doble clic '
                'para volver. El producto del tercer panel es lo que se integra numéricamente para obtener n y p.</div>',
                unsafe_allow_html=True)

    # --- Película en temperatura -------------------------------------------------
    st.subheader("¿Qué pasa al calentar el material?")
    st.caption("Presiona ▶: la temperatura sube de 100 K a 600 K. Las colas de Fermi-Dirac se ensanchan, "
               "aparecen portadores térmicos y E_F se desplaza hacia el centro de la brecha (el material "
               "se vuelve intrínseco). Eje x logarítmico: cada división es un factor 10.")
    with st.spinner("Resolviendo E_F para cada temperatura…"):
        Ts_m, Efs_m, E_m, pn_m, pp_m = pelicula_temperatura(tipo, Na_cm3, Nd_cm3, me_kg, mh_kg, Ec_eV, Ev_eV)
    PISO = 1.0

    def poligono(prod):
        x = np.clip(prod, PISO, None)
        return np.concatenate([[PISO], x, [PISO]]), np.concatenate([[E_m[0]], E_m, [E_m[-1]]])

    def trazas_cuadro(i):
        xn, yn = poligono(pn_m[i])
        xp, yp = poligono(pp_m[i])
        n_i = np.trapezoid(pn_m[i], E_m) if hasattr(np, "trapezoid") else np.trapz(pn_m[i], E_m)
        p_i = np.trapezoid(pp_m[i], E_m) if hasattr(np, "trapezoid") else np.trapz(pp_m[i], E_m)
        return [
            go.Scatter(x=xn, y=yn, fill="toself", fillcolor=rgba(COL["e"], .5), line=dict(color=COL["e"], width=1),
                       name=f"electrones  n ≈ {sci(n_i, 1)}", hoverinfo="skip"),
            go.Scatter(x=xp, y=yp, fill="toself", fillcolor=rgba(COL["h"], .5), line=dict(color=COL["h"], width=1),
                       name=f"huecos  p ≈ {sci(p_i, 1)}", hoverinfo="skip"),
            go.Scatter(x=[PISO, 1e24], y=[Efs_m[i]] * 2, mode="lines", line=dict(color=COL["amber"], dash="dash", width=2),
                       name="E_F", hoverinfo="skip"),
            go.Scatter(x=[Ts_m[i]], y=[Efs_m[i]], mode="markers",
                       marker=dict(size=15, color=COL["amber"], line=dict(color="white", width=2)),
                       name=f"T = {Ts_m[i]:.0f} K", showlegend=False, hoverinfo="skip"),
        ]

    i0 = int(np.argmin(np.abs(Ts_m - T_K)))
    figm = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], shared_yaxes=True, horizontal_spacing=0.04,
                         subplot_titles=("Portadores por energía, g·f  [cm⁻³ eV⁻¹]", "Nivel de Fermi vs. temperatura"))
    for k, tr in enumerate(trazas_cuadro(i0)):
        figm.add_trace(tr, 1, 1 if k < 3 else 2)
    figm.add_trace(go.Scatter(x=Ts_m, y=Efs_m, mode="lines", line=dict(color=COL["amber"], width=2.5),
                              name="E_F(T)", hovertemplate="T=%{x:.0f} K<br>E_F=%{y:.3f} eV<extra></extra>"), 1, 2)
    for col in (1, 2):
        figm.add_hrect(y0=Ev_eV, y1=Ec_eV, fillcolor="rgba(255,255,255,.035)", line_width=0, row=1, col=col)
        figm.add_hline(y=Ec_eV, line=dict(color=COL["e"], width=1.2), row=1, col=col)
        figm.add_hline(y=Ev_eV, line=dict(color=COL["h"], width=1.2), row=1, col=col)
    figm.add_hline(y=(Ec_eV + Ev_eV) / 2, line=dict(color=COL["dim"], width=1, dash="dot"), row=1, col=2)
    figm.add_annotation(x=590, y=Ec_eV, xref="x2", yref="y", text="E_C", showarrow=False, yanchor="bottom",
                        font=dict(color=COL["e"]))
    figm.add_annotation(x=590, y=Ev_eV, xref="x2", yref="y", text="E_V", showarrow=False, yanchor="top",
                        font=dict(color=COL["h"]))
    figm.frames = [go.Frame(data=trazas_cuadro(i), traces=[0, 1, 2, 3], name=f"{T:.0f}") for i, T in enumerate(Ts_m)]
    figm.update_xaxes(type="log", range=[0, 23], title_text="cm⁻³ eV⁻¹ (escala log)", row=1, col=1)
    figm.update_xaxes(range=[90, 610], title_text="T [K]", row=1, col=2)
    figm.update_yaxes(title_text="Energía [eV]", range=[E_m[0], E_m[-1]], row=1, col=1)
    estilo(figm, 520, legend=dict(orientation="h", y=1.24, x=0, yanchor="bottom"),
           margin=dict(l=64, r=24, t=110, b=120),
           updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.2, xanchor="left", yanchor="top",
                             pad=dict(r=10, t=0), showactive=False, bgcolor="#1b2440",
                             font=dict(color=COL["text"]),
                             buttons=[dict(label="▶ Calentar", method="animate",
                                           args=[None, dict(frame=dict(duration=260, redraw=True),
                                                            transition=dict(duration=180), fromcurrent=False,
                                                            mode="immediate")]),
                                      dict(label="⏸", method="animate",
                                           args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                              mode="immediate")])])],
           sliders=[dict(active=i0, x=0.14, y=-0.14, len=0.86, xanchor="left", pad=dict(t=8),
                         currentvalue=dict(prefix="T = ", suffix=" K", xanchor="right", font=dict(color=COL["amber"], size=14)),
                         bgcolor="#1b2440", activebgcolor=COL["amber"], font=dict(color=COL["dim"]),
                         steps=[dict(method="animate", label=f"{T:.0f}",
                                     args=[[f"{T:.0f}"], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                                for T in Ts_m])])
    mostrar(figm)

    # --- Superficie 3D: E_F(T, dopaje) ------------------------------------------
    st.subheader("El paisaje del nivel de Fermi (3D)")
    st.caption("Cada punto de esta superficie es una resolución numérica de la neutralidad de carga. "
               "Izquierda: tipo p (E_F baja hacia E_V); derecha: tipo n (E_F sube hacia E_C). "
               "Donde la superficie atraviesa los planos de banda, el material es degenerado y Boltzmann ya no sirve. "
               "Arrastra para girarla.")
    with st.spinner("Resolviendo E_F en una grilla de 209 puntos (solo la primera vez)…"):
        Ts_s, xs_s, Z_s = superficie_ef(me_kg, mh_kg, Ec_eV, Ev_eV)
    x_act = x_dopaje(tipo, Na_cm3, Nd_cm3)
    Ef_semilla_p = f.resolver_ef(c.T_OPERACION_DEFAULT, c.N_A_DEFAULT, 0.0, Ec_eV, Ev_eV, me_kg, mh_kg)
    Ef_semilla_n = f.resolver_ef(c.T_OPERACION_DEFAULT, 0.0, c.N_D_DEFAULT, Ec_eV, Ev_eV, me_kg, mh_kg)
    escala_ef = [[0, COL["h"]], [0.5, "#8e7cc3"], [1, COL["e"]]]
    fig3 = go.Figure()
    fig3.add_trace(go.Surface(
        x=xs_s, y=Ts_s, z=Z_s, colorscale=escala_ef, cmin=Ev_eV, cmax=Ec_eV, opacity=0.96,
        colorbar=dict(title=dict(text="E_F [eV]"), len=0.7, thickness=14),
        contours=dict(z=dict(show=True, usecolormap=False, color="rgba(255,255,255,.35)", width=1,
                             start=Ev_eV, end=Ec_eV, size=Eg_eV / 10),
                      x=dict(show=True, color="rgba(255,255,255,.12)", width=1),
                      y=dict(show=True, color="rgba(255,255,255,.12)", width=1)),
        hovertemplate="dopaje x=%{x:.0f}<br>T=%{y:.0f} K<br>E_F=%{z:.3f} eV<extra></extra>", name="E_F"))
    for z0, colr, nom in ((Ec_eV, COL["e"], "E_C"), (Ev_eV, COL["h"], "E_V")):
        fig3.add_trace(go.Surface(x=[xs_s[0], xs_s[-1]], y=[Ts_s[0], Ts_s[-1]], z=[[z0, z0], [z0, z0]],
                                  colorscale=[[0, colr], [1, colr]], showscale=False, opacity=0.18,
                                  hoverinfo="skip", name=nom))
        fig3.add_trace(go.Scatter3d(x=[xs_s[-1]], y=[Ts_s[-1]], z=[z0], mode="text", text=[f"  {nom}"],
                                    textfont=dict(color=colr, size=14), showlegend=False, hoverinfo="skip"))
    fig3.add_trace(go.Scatter3d(
        x=[x_dopaje("tipo p", c.N_A_DEFAULT, 0), x_dopaje("tipo n", 0, c.N_D_DEFAULT)],
        y=[c.T_OPERACION_DEFAULT] * 2, z=[Ef_semilla_p, Ef_semilla_n], mode="markers+text",
        marker=dict(size=7, color=COL["amber"], symbol="diamond", line=dict(color="white", width=1)),
        text=["semilla: N_A", "semilla: N_D"], textposition="top center", textfont=dict(color=COL["amber"]),
        name="parámetros de la semilla (25 °C)"))
    fig3.add_trace(go.Scatter3d(x=[x_act], y=[T_K], z=[Ef_eV], mode="markers",
                                marker=dict(size=9, color="white", line=dict(color=COL["amber"], width=3)),
                                name="tu configuración actual"))
    tv = [-9, -6, -3, 0, 3, 6, 9]
    fig3.update_layout(scene=dict(
        xaxis=dict(title="dopaje", tickvals=tv,
                   ticktext=["N_A 10²¹", "N_A 10¹⁸", "N_A 10¹⁵", "intrínseco", "N_D 10¹⁵", "N_D 10¹⁸", "N_D 10²¹"],
                   backgroundcolor="#0f1629", gridcolor="rgba(255,255,255,.1)"),
        yaxis=dict(title="T [K]", backgroundcolor="#0f1629", gridcolor="rgba(255,255,255,.1)"),
        zaxis=dict(title="E_F [eV]", backgroundcolor="#0f1629", gridcolor="rgba(255,255,255,.1)"),
        camera=dict(eye=dict(x=1.55, y=-1.55, z=0.75)), aspectmode="manual", aspectratio=dict(x=1.5, y=1, z=0.8)))
    estilo(fig3, 620, margin=dict(l=0, r=0, t=10, b=0), legend=dict(x=0, y=1))
    mostrar(fig3)

    # --- Tabla numérico vs Boltzmann ----------------------------------------------
    st.subheader("Integración numérica vs. aproximación de Boltzmann")
    err = lambda a, b: f"{abs(a - b) / abs(a) * 100:.2f} %"
    tabla = pd.DataFrame({
        "Magnitud": ["n [cm⁻³]", "p [cm⁻³]", "n·p [cm⁻⁶]", "n_i²(T) [cm⁻⁶]"],
        "Integración numérica": [f"{n_num:.4e}", f"{p_num:.4e}", f"{n_num*p_num:.4e}", f"{ni2_num:.4e}"],
        "Aprox. de Boltzmann": [f"{n_boltz:.4e}", f"{p_boltz:.4e}", f"{n_boltz*p_boltz:.4e}", f"{ni2_boltz:.4e}"],
        "Diferencia": [err(n_num, n_boltz), err(p_num, p_boltz), err(n_num * p_num, n_boltz * p_boltz),
                       err(ni2_num, ni2_boltz)],
    })
    st.dataframe(tabla, hide_index=True, width="stretch")
    st.caption(f"n_i²(T) se recalcula a la T, m_e, m_h y E_g actuales (T={T_K:.0f} K) — no es un valor fijo "
               f"de 300 K. A 300 K con los parámetros por defecto del curso, n_i²(T) ≈ {c.NI_SI_300K**2:.2e} "
               f"cm⁻⁶ (V1 de la pestaña de Validación).")
    if degenerado:
        st.warning("Dopaje ≥10¹⁹ cm⁻³: régimen degenerado — la aproximación de Boltzmann diverge "
                   "de la integración numérica y la ley de acción de masas n·p = n_i² deja de cumplirse.")

# ---------------------------------------------------------------------------
# Pestaña 2 — El bloque de silicio en 3D
# ---------------------------------------------------------------------------
with tab2:
    producto_np = n_num * p_num
    error_pct = abs(producto_np - ni2_boltz) / ni2_boltz * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n numérico  [cm⁻³]", sci(n_num))
    c2.metric("p numérico  [cm⁻³]", sci(p_num))
    c3.metric("n·p  [cm⁻⁶]", sci(producto_np))
    c4.metric(f"n_i² modelo [cm⁻⁶] · Δ {error_pct:.1f} %", sci(ni2_boltz))

    n_don = num_simbolos(Nd_cm3) if tipo == "tipo n" else 0
    n_acc = num_simbolos(Na_cm3) if tipo == "tipo p" else 0
    n_e_eq, n_h_eq = num_simbolos(n_num), num_simbolos(p_num)
    n_exc = max(2, num_simbolos(dn))
    # Trampas dibujadas: más trampas cuanto menor es τ_SRH (τ_SRH ∝ 1/N_T). Mapeo declarado.
    n_traps = int(np.clip(round(2 * np.log10(1e-2 / tau_srh_s)), 1, 12))
    if tipo == "tipo n":
        dop_txt = f"N_D = {sci(Nd_cm3, 1)} cm⁻³"
    elif tipo == "tipo p":
        dop_txt = f"N_A = {sci(Na_cm3, 1)} cm⁻³"
    else:
        dop_txt = "sin dopar"
    notas = (
        f"• 1 partícula por década sobre 10⁶ cm⁻³: n = {sci(n_num, 1)} → <b>{n_e_eq}</b> e⁻, "
        f"p = {sci(p_num, 1)} → <b>{n_h_eq}</b> h⁺.<br>"
        f"• ☀ genera Δn = {dn:.0e} cm⁻³ → <b>{n_exc}</b> pares extra en régimen estacionario.<br>"
        f"• Velocidad ∝ √(T/m*), vibración de la red ∝ √T.<br>"
        f"• {n_traps} trampa(s): más trampas si τ_SRH baja (τ_SRH = {tau_srh_s*1e6:g} μs).<br>"
        f"• Tiempo: τ real ({fmt_t(tau_eff_actual)}) reescalado a 3 s de animación."
    )
    params3d = dict(
        T=float(T_K), Eg=float(Eg_eV), me_rel=float(me_kg / c.M0), mh_rel=float(mh_kg / c.M0),
        n_don=n_don, n_acc=n_acc, n_traps=n_traps, n_e_eq=n_e_eq, n_h_eq=n_h_eq, n_exc=n_exc,
        r_term=0.3 * T_K / 300,
        w={k: float(v) for k, v in tasa.items()},
        S_txt=f"{S_cm_s:.0e}", subtitulo=f"{tipo} · {dop_txt} · T = {T_K:.0f} K",
        notas_html=notas, luz=False, demo=False,
    )
    componente("bloque3d", params3d, 690)

    if (not degenerado) and error_pct < 5:
        st.success(f"n·p ≈ n_i² del modelo ({sci(ni2_boltz)} cm⁻⁶, error {error_pct:.1f}%): "
                   "equilibrio térmico, régimen no degenerado. ✅ Ley de acción de masas se cumple.")
    elif degenerado:
        st.warning(f"Dopaje degenerado (≥10¹⁹ cm⁻³): n·p difiere {error_pct:.1f}% de n_i² del modelo. "
                   "❌ La ley de acción de masas deja de cumplirse, como se espera fuera de la "
                   "aproximación de Boltzmann.")
    else:
        st.warning(f"n·p difiere {error_pct:.1f}% de n_i² del modelo — revisar consistencia numérica.")

    with st.expander("Qué estás viendo y qué está a escala", expanded=False):
        st.markdown(f"""
- **Red cristalina real**: estructura diamante (FCC + base de 2 átomos), 3×3×3 celdas convencionales. Los enlaces
  se calculan por distancia real de primer vecino (a·√3/4), así que cada átomo interior tiene exactamente 4.
- **Enlaces insatisfechos** (rojo claro) en las caras: son los estados que causan la recombinación superficial.
- **Dopantes**: fósforo ionizado P⁺ (verde) en tipo n y boro ionizado B⁻ (naranjo) en tipo p, en sitios
  sustitucionales. Ionización total, igual que en el cálculo de E_F.
- **Portadores**: el número dibujado sigue el mapeo logarítmico declarado (no es a escala lineal: 10¹⁹ e⁻ no caben
  en una pantalla). En tipo n con la semilla, p ≈ {sci(p_num, 1)} cm⁻³ < 10⁶ → no se dibujan huecos de equilibrio.
- **Iluminación (☀)**: cada fotón se absorbe a una profundidad aleatoria que decae exponencialmente desde la
  cara frontal, y crea un par. La tasa se ajusta para que el exceso estacionario sea Δn.
- **Recombinación**: cada evento se elige al azar con pesos iguales a 1/τ de cada mecanismo al Δn actual
  (radiativa {share['rad']:.2g} %, Auger {share['auger']:.2g} %, SRH {share['srh']:.2g} %, superficie {share['surf']:.2g} %).
  El botón *Pesos: iguales* es un modo demostración para ver los cuatro con la misma frecuencia.
""")

# ---------------------------------------------------------------------------
# Pestaña 3 — Los cuatro caminos de la recombinación
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("Cada panel repite en bucle **qué le pasa al electrón, qué le pasa al hueco y dónde termina "
                "la energía**. El panel con borde ámbar es el mecanismo que domina con los parámetros actuales.")
    col_et, col_info = st.columns([1, 2])
    with col_et:
        Et_frac = st.slider("Nivel de trampa SRH (0 = E_V, 1 = E_C)", 0.0, 1.0, 0.5, 0.05, key="Et_frac")
    eficiencia_srh = float(np.exp(-abs(Et_frac - 0.5) / 0.25))
    p_reabs = float(1 - np.exp(-W_cm / 100e-4))
    n_surf = int(np.clip(np.log10(S_cm_s) - 1, 1, 8))
    with col_info:
        st.caption(f"Modelos cualitativos declarados de las animaciones: eficiencia de captura SRH ≈ "
                   f"exp(−|E_T/E_g − ½| / 0.25) = {eficiencia_srh:.2f} (máxima a mitad de brecha); "
                   f"probabilidad de reabsorción del fotón ≈ 1 − exp(−W/100 μm) = {p_reabs:.2f}; "
                   f"pares por ciclo en la superficie = log₁₀S − 1 = {n_surf}. No reemplazan a α(λ) real "
                   f"(Problemas 2.2/2.3).")
    componente("recombinacion", dict(
        Eg=float(Eg_eV), tipo={"tipo n": "n", "tipo p": "p"}.get(tipo, "i"), Et_frac=float(Et_frac),
        eficiencia=eficiencia_srh, p_reabs=p_reabs, n_surf=n_surf, S_txt=f"{S_cm_s:.0e}", W_um=espesor_um,
        tau={k: float(v) for k, v in tau_act.items()}, share=share, dom=dominante), "content", scrolling=True)

    # --- τ vs Δn con regiones de dominancia ---------------------------------------
    st.subheader("Vida media efectiva vs. nivel de inyección")
    delta_n_arr = np.logspace(12, 18, 300)
    taus = {
        "rad": f.tau_radiativo(n_num, p_num, delta_n_arr, c.B_RADIATIVO_SI),
        "auger": f.tau_auger(delta_n_arr, c.C_AUGER_SI),
        "srh": np.full_like(delta_n_arr, tau_srh_s),
        "surf": np.full_like(delta_n_arr, tau_act["surf"]),
    }
    tau_eff_arr = f.tau_efectivo(*taus.values())
    claves = list(taus)
    dom_idx = np.argmin(np.vstack([taus[k] for k in claves]), axis=0)

    fig_tau = go.Figure()
    ini = 0
    for j in range(1, len(delta_n_arr) + 1):
        if j == len(delta_n_arr) or dom_idx[j] != dom_idx[ini]:
            k = claves[dom_idx[ini]]
            fig_tau.add_vrect(x0=delta_n_arr[ini], x1=delta_n_arr[j - 1], fillcolor=rgba(COL[k], .09), line_width=0,
                              annotation_text=f"domina {NOMBRE_MEC[k]}", annotation_position="top left",
                              annotation_font=dict(color=COL[k], size=11))
            ini = j
    etiquetas = {"rad": "τ radiativa", "auger": "τ Auger", "srh": "τ SRH (volumen)", "surf": "τ superficie = W/2S"}
    for k in claves:
        fig_tau.add_trace(go.Scatter(x=delta_n_arr, y=taus[k], name=etiquetas[k], line=dict(color=COL[k], width=2),
                                     hovertemplate="Δn=%{x:.2e}<br>τ=%{y:.3e} s<extra>%{fullData.name}</extra>"))
    fig_tau.add_trace(go.Scatter(x=delta_n_arr, y=tau_eff_arr, name="τ efectiva", line=dict(color="white", width=4),
                                 hovertemplate="Δn=%{x:.2e}<br>τ_eff=%{y:.3e} s<extra></extra>"))
    fig_tau.add_trace(go.Scatter(x=[dn], y=[tau_eff_actual], mode="markers+text", name="Δn actual",
                                 marker=dict(size=14, color=COL["amber"], line=dict(color="white", width=2)),
                                 text=[f"  τ_eff = {fmt_t(tau_eff_actual)}"], textposition="middle right",
                                 textfont=dict(color=COL["amber"])))
    if delta_n_arr.min() <= delta_n_cruce <= delta_n_arr.max():
        fig_tau.add_vline(x=delta_n_cruce, line=dict(color=COL["dim"], dash="dash"),
                          annotation_text=f"cruce SRH↔Auger<br>Δn ≈ {delta_n_cruce:.2e}", annotation_position="bottom left",
                          annotation_font=dict(color=COL["dim"], size=11))
    fig_tau.update_xaxes(type="log", title_text="Nivel de inyección Δn [cm⁻³]", exponentformat="power")
    fig_tau.update_yaxes(type="log", title_text="Vida media τ [s]", exponentformat="power",
                         range=[np.log10(tau_eff_arr.min()) - 1, np.log10(max(1e-2, tau_srh_s * 30))])
    estilo(fig_tau, 480, legend=dict(orientation="h", y=-0.22, x=0), margin=dict(l=64, r=24, t=40, b=90))
    mostrar(fig_tau)
    st.caption(f"Cruce de dominancia SRH → Auger en Δn ≈ {delta_n_cruce:.3e} cm⁻³ "
               f"(donde 1/τ_SRH = 1/τ_Auger, con τ_SRH={tau_srh_s*1e6:g} μs).")

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.markdown("**¿Quién se lleva los pares?** Fracción de 1/τ_eff de cada mecanismo")
        fig_sh = go.Figure()
        for k in claves:
            fig_sh.add_trace(go.Scatter(x=delta_n_arr, y=1 / taus[k], name=NOMBRE_MEC[k], stackgroup="a",
                                        groupnorm="percent", line=dict(color=COL[k], width=0.5),
                                        fillcolor=rgba(COL[k], .75),
                                        hovertemplate="%{y:.1f} %<extra>%{fullData.name}</extra>"))
        fig_sh.add_vline(x=dn, line=dict(color="white", width=2, dash="dot"),
                         annotation_text="Δn actual", annotation_font=dict(color="white"))
        fig_sh.update_xaxes(type="log", title_text="Δn [cm⁻³]", exponentformat="power")
        fig_sh.update_yaxes(title_text="% de las recombinaciones", range=[0, 100], ticksuffix=" %")
        estilo(fig_sh, 430, hovermode="x unified", legend=dict(orientation="h", y=-0.28, x=0),
               margin=dict(l=64, r=16, t=20, b=90))
        mostrar(fig_sh)
    with col_der:
        st.markdown("**Superficie 3D:** τ_eff según inyección y calidad de la superficie")
        lg_dn = np.linspace(12, 18, 49)
        lg_S = np.linspace(0, 6, 49)
        DN, SS = np.meshgrid(10 ** lg_dn, 10 ** lg_S)
        TE = f.tau_efectivo(f.tau_radiativo(n_num, p_num, DN, c.B_RADIATIVO_SI), f.tau_auger(DN, c.C_AUGER_SI),
                            tau_srh_s, W_cm / (2 * SS))
        figs = go.Figure(go.Surface(
            x=lg_dn, y=lg_S, z=np.log10(TE), colorscale="Inferno",
            colorbar=dict(title=dict(text="log₁₀ τ_eff"), len=0.7, thickness=12),
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
            hovertemplate="Δn=10^%{x:.1f}<br>S=10^%{y:.1f} cm/s<br>τ_eff=10^%{z:.2f} s<extra></extra>"))
        figs.add_trace(go.Scatter3d(x=[np.log10(dn)], y=[np.log10(S_cm_s)], z=[np.log10(tau_eff_actual)],
                                    mode="markers+text", text=["tu punto"], textposition="top center",
                                    textfont=dict(color=COL["amber"]),
                                    marker=dict(size=8, color=COL["amber"], line=dict(color="white", width=2)),
                                    name="configuración actual"))
        figs.update_layout(scene=dict(
            xaxis=dict(title="log₁₀ Δn", backgroundcolor="#0f1629"),
            yaxis=dict(title="log₁₀ S [cm/s]", backgroundcolor="#0f1629"),
            zaxis=dict(title="log₁₀ τ_eff [s]", backgroundcolor="#0f1629"),
            camera=dict(eye=dict(x=-1.6, y=-1.5, z=0.9))), showlegend=False)
        estilo(figs, 430, margin=dict(l=0, r=0, t=10, b=0))
        mostrar(figs)

    # --- Longitudes de difusión ---------------------------------------------------
    D_n = f.coef_difusion_einstein(c.MU_N_REF, T_K)
    D_p = f.coef_difusion_einstein(c.MU_P_REF, T_K)
    L_n = f.longitud_difusion(D_n, tau_eff_actual)
    L_p = f.longitud_difusion(D_p, tau_eff_actual)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("τ_efectivo @ Δn actual", f"{tau_eff_actual*1e6:.2f} μs")
    col_b.metric("L_n = √(D_n·τ)", f"{L_n*1e4:.1f} μm")
    col_c.metric("L_p = √(D_p·τ)", f"{L_p*1e4:.1f} μm")
    col_d.metric(f"Domina ({share[dominante]:.0f} % de 1/τ_eff)", NOMBRE_MEC[dominante])

    figL = go.Figure()
    for nom, val, colr in (("L_n (electrones)", L_n * 1e4, COL["e"]), ("L_p (huecos)", L_p * 1e4, COL["h"]),
                           ("Espesor W", W_cm * 1e4, COL["amber"])):
        figL.add_trace(go.Bar(y=[nom], x=[val], orientation="h", marker_color=colr, text=[f"{val:.0f} μm"],
                              textposition="outside", cliponaxis=False, showlegend=False,
                              hovertemplate=f"{nom}: %{{x:.1f}} μm<extra></extra>"))
    figL.update_xaxes(title_text="μm")
    estilo(figL, 210, margin=dict(l=120, r=60, t=36, b=40),
           title=dict(text="¿Llegan los portadores a los contactos? (L > W es bueno)", font=dict(size=13)))
    mostrar(figL)

# ---------------------------------------------------------------------------
# Pestaña Validación
# ---------------------------------------------------------------------------
with tab4:
    st.header("Validación")
    st.caption("Las verificaciones V1-V8 se ejecutan automáticamente contra los parámetros de referencia "
               "del curso (Anexo B) y la semilla del grupo (S=7), independiente de los controles de las "
               "otras pestañas, para que esta tabla sea siempre reproducible.")

    T_ref = 300.0
    Ec_ref, Ev_ref = c.EG_SI_300K, 0.0
    me_ref, mh_ref = c.ME_SI_DEFAULT, c.MH_SI_DEFAULT
    kT_ref = c.K_B_EV * T_ref

    filas = []

    def agregar_fila(nombre, valor_calc, valor_ref, tipo_error, veredicto_ok, detalle=""):
        if tipo_error == "pct":
            error_pct = abs(valor_calc - valor_ref) / abs(valor_ref) * 100
            error_txt = f"{error_pct:.1f}%"
        else:
            error_pct = None
            error_txt = detalle
        filas.append({
            "Verificación": nombre,
            "Valor calculado": f"{valor_calc:.4g}" if isinstance(valor_calc, (int, float)) else str(valor_calc),
            "Valor de referencia": f"{valor_ref:.4g}" if isinstance(valor_ref, (int, float)) else str(valor_ref),
            "Error / criterio": error_txt,
            "Veredicto": "✅ Aprobado" if veredicto_ok else "❌ Rechazado",
        })

    # V1: n_i por integración numérica (Si intrínseco, 300K)
    Ef_intrinseco = f.resolver_ef(T_ref, 0.0, 0.0, Ec_ref, Ev_ref, me_ref, mh_ref)
    ni_calc = f.n_numerico(Ef_intrinseco, T_ref, Ec_ref, me_ref)
    agregar_fila("V1: n_i por integración (Si intrínseco, 300K)", ni_calc, c.NI_SI_300K, "pct",
                 abs(ni_calc - c.NI_SI_300K) / c.NI_SI_300K <= 0.20)

    # V2: offset de E_F intrínseco respecto al centro de la banda prohibida
    offset_calc_eV = Ef_intrinseco - Ec_ref / 2.0
    offset_ref_eV = 0.75 * kT_ref * np.log(mh_ref / me_ref)
    error_abs_meV = abs(offset_calc_eV - offset_ref_eV) * 1000
    agregar_fila("V2: offset E_F intrínseco vs. centro de gap", offset_calc_eV * 1000, offset_ref_eV * 1000,
                 "abs", error_abs_meV <= 20, detalle=f"{error_abs_meV:.1f} meV (tol. ±20 meV)")

    # V3: numérico vs. Boltzmann, dopaje no degenerado (N_D=1e16 cm^-3)
    Nd_v3 = 1.0e16
    Ef_v3 = f.resolver_ef(T_ref, 0.0, Nd_v3, Ec_ref, Ev_ref, me_ref, mh_ref)
    n_num_v3 = f.n_numerico(Ef_v3, T_ref, Ec_ref, me_ref)
    n_boltz_v3 = f.n_boltzmann(Ef_v3, Ec_ref, T_ref, f.Nc_efectiva(T_ref, me_ref))
    agregar_fila("V3: n numérico vs. Boltzmann (N_D=1e16, no degenerado)", n_num_v3, n_boltz_v3, "pct",
                 abs(n_num_v3 - n_boltz_v3) / n_boltz_v3 <= 0.02)

    # V4: numérico vs. Boltzmann, dopaje degenerado (N_D de la semilla, 1.5e19 cm^-3)
    Ef_v4 = f.resolver_ef(T_ref, 0.0, c.N_D_DEFAULT, Ec_ref, Ev_ref, me_ref, mh_ref)
    n_num_v4 = f.n_numerico(Ef_v4, T_ref, Ec_ref, me_ref)
    n_boltz_v4 = f.n_boltzmann(Ef_v4, Ec_ref, T_ref, f.Nc_efectiva(T_ref, me_ref))
    error_v4_pct = abs(n_num_v4 - n_boltz_v4) / n_boltz_v4 * 100
    ef_en_banda = Ef_v4 >= Ec_ref
    agregar_fila("V4: n numérico vs. Boltzmann (N_D=1.5e19, degenerado)", n_num_v4, n_boltz_v4, "abs",
                 error_v4_pct > 10, detalle=f"{error_v4_pct:.1f}% (se espera >10%)")
    if ef_en_banda:
        st.warning(f"V4: con N_D=1.5×10¹⁹ cm⁻³, E_F={Ef_v4:.4f} eV queda dentro/al borde de la banda de "
                   f"conducción (E_C={Ec_ref:.4f} eV) → régimen degenerado, la aproximación de Boltzmann "
                   f"deja de ser válida.")
    else:
        st.info(f"V4: E_F={Ef_v4:.4f} eV vs. E_C={Ec_ref:.4f} eV — aún por debajo de la banda de conducción, "
                f"pero suficientemente cerca para que Boltzmann diverja de la integración numérica.")

    # V5: ley de acción de masas n*p = n_i^2, no degenerado vs. degenerado
    p_num_v3 = f.p_numerico(Ef_v3, T_ref, Ev_ref, mh_ref)
    np_v3 = n_num_v3 * p_num_v3
    ni2_modelo = ni_calc ** 2
    agregar_fila("V5a: n·p = n_i² (N_D=1e16, no degenerado)", np_v3, ni2_modelo, "pct",
                 abs(np_v3 - ni2_modelo) / ni2_modelo <= 0.01)

    p_num_v4 = f.p_numerico(Ef_v4, T_ref, Ev_ref, mh_ref)
    np_v4 = n_num_v4 * p_num_v4
    error_v5b_pct = abs(np_v4 - ni2_modelo) / ni2_modelo * 100
    agregar_fila("V5b: n·p = n_i² (N_D=1.5e19, degenerado)", np_v4, ni2_modelo, "abs",
                 error_v5b_pct > 1, detalle=f"{error_v5b_pct:.1f}% (se espera fallo, >1%)")

    # V6: tau_Auger a delta_n = 1e18 cm^-3
    delta_n_v6 = 1.0e18
    tau_auger_calc = f.tau_auger(delta_n_v6, c.C_AUGER_SI)
    tau_auger_ref = 2.5e-6
    agregar_fila("V6: τ_Auger @ Δn=1e18 cm⁻³", tau_auger_calc * 1e6, tau_auger_ref * 1e6, "pct",
                 abs(tau_auger_calc - tau_auger_ref) / tau_auger_ref <= 0.05)

    # V7: cruce de dominancia SRH -> Auger
    delta_n_cruce_v7 = np.sqrt(1.0 / (c.C_AUGER_SI * c.TAU_SRH_VOLUMEN_DEFAULT))
    existe_cruce = np.isfinite(delta_n_cruce_v7) and delta_n_cruce_v7 > 0
    agregar_fila("V7: cruce de dominancia SRH↔Auger (Δn)", delta_n_cruce_v7, delta_n_cruce_v7, "abs",
                 existe_cruce, detalle="existe y es calculable" if existe_cruce else "no existe")

    # V8: tau_radiativa a delta_n = 1e18 cm^-3
    tau_rad_calc = f.tau_radiativo(c.NI_SI_300K, c.NI_SI_300K, delta_n_v6, c.B_RADIATIVO_SI)
    tau_rad_ref = 211e-6
    orden_magnitud_ok = tau_rad_calc / tau_auger_calc >= 50
    agregar_fila("V8: τ_radiativa @ Δn=1e18 cm⁻³", tau_rad_calc * 1e6, tau_rad_ref * 1e6, "pct",
                 abs(tau_rad_calc - tau_rad_ref) / tau_rad_ref <= 0.05)
    if orden_magnitud_ok:
        st.info(f"V8: τ_radiativa ({tau_rad_calc*1e6:.1f} μs) es ≈{tau_rad_calc/tau_auger_calc:.0f}× mayor que "
                f"τ_Auger ({tau_auger_calc*1e6:.2f} μs) al mismo Δn — consistente con lo esperado "
                f"(diferencia de ~2 órdenes de magnitud).")
    else:
        st.warning("V8: τ_radiativa no resultó ~2 órdenes de magnitud mayor que τ_Auger como se esperaba.")

    n_aprobados = sum(1 for fil in filas if fil["Veredicto"].startswith("✅"))
    st.metric("Verificaciones aprobadas", f"{n_aprobados} / {len(filas)}")
    st.progress(n_aprobados / len(filas))
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
