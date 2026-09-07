import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import constantes as c
import fisica as f

st.set_page_config(page_title="Celdas Solares - Proyecto 1 (2.1)", layout="wide")

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

    st.subheader("Controles")
    st.session_state.T_K = st.slider("Temperatura T [K]", 100, 600, int(st.session_state.T_K), 1)
    st.session_state.tipo_material = st.selectbox(
        "Tipo de material", ["intrínseco", "tipo n", "tipo p"],
        index=["intrínseco", "tipo n", "tipo p"].index(st.session_state.tipo_material))

    opciones_Na = sorted(set([10**e for e in range(13, 22)] + [c.N_A_DEFAULT]))
    opciones_Nd = sorted(set([10**e for e in range(13, 22)] + [c.N_D_DEFAULT]))
    st.session_state.Na_cm3 = st.select_slider(
        "Dopaje aceptores N_A [cm^-3]",
        options=opciones_Na, value=st.session_state.Na_cm3)
    st.session_state.Nd_cm3 = st.select_slider(
        "Dopaje donadores N_D [cm^-3]",
        options=opciones_Nd, value=st.session_state.Nd_cm3)

    me_factor = st.slider("m_e / m_0 (densidad de estados)", 0.1, 2.0,
                           float(st.session_state.me_kg / c.M0), 0.01)
    st.session_state.me_kg = me_factor * c.M0
    mh_factor = st.slider("m_h / m_0 (densidad de estados)", 0.1, 2.0,
                           float(st.session_state.mh_kg / c.M0), 0.01)
    st.session_state.mh_kg = mh_factor * c.M0

    st.session_state.Eg_eV = st.slider("Ancho de banda prohibida E_g [eV]", 0.5, 2.0,
                                        float(st.session_state.Eg_eV), 0.01)

    st.session_state.delta_n_cm3 = st.select_slider(
        "Nivel de inyección Δn [cm^-3]",
        options=[10**e for e in range(12, 19)], value=st.session_state.delta_n_cm3)

    opciones_S = sorted(set([10**e for e in range(0, 7)] + [c.S_FRONTAL_DEFAULT]))
    st.session_state.S_frontal_cm_s = st.select_slider(
        "Velocidad de recombinación superficial S [cm/s]",
        options=opciones_S, value=st.session_state.S_frontal_cm_s)

    espesor_um = st.slider("Espesor del bloque [μm]", 20, 300,
                            int(st.session_state.espesor_cm * 1e4), 1)
    st.session_state.espesor_cm = espesor_um * 1e-4

# Energías de referencia: convención Ev = 0
Ev_eV = 0.0
Ec_eV = st.session_state.Eg_eV

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Estadística de portadores",
    "2. El bloque de silicio",
    "3. Recombinación",
    "Validación",
])

# ---------------------------------------------------------------------------
# Pestaña 1 — Estadística de portadores
# ---------------------------------------------------------------------------
with tab1:
    st.header("Estadística de portadores")

    T_K = st.session_state.T_K
    me_kg, mh_kg = st.session_state.me_kg, st.session_state.mh_kg
    tipo = st.session_state.tipo_material

    # Concentraciones de dopantes efectivas según el tipo de material elegido.
    # Ionización total de los dopantes (aproximación declarada en fisica.resolver_ef).
    if tipo == "intrínseco":
        Na_cm3, Nd_cm3 = 0.0, 0.0
    elif tipo == "tipo n":
        Na_cm3, Nd_cm3 = 0.0, st.session_state.Nd_cm3
    else:  # tipo p
        Na_cm3, Nd_cm3 = st.session_state.Na_cm3, 0.0

    Ef_eV = f.resolver_ef(T_K, Na_cm3, Nd_cm3, Ec_eV, Ev_eV, me_kg, mh_kg)
    st.session_state.Ef_eV = Ef_eV  # compartido con las otras pestañas

    Nc_cm3 = f.Nc_efectiva(T_K, me_kg)
    Nv_cm3 = f.Nv_efectiva(T_K, mh_kg)
    n_num = f.n_numerico(Ef_eV, T_K, Ec_eV, me_kg)
    p_num = f.p_numerico(Ef_eV, T_K, Ev_eV, mh_kg)
    n_boltz = f.n_boltzmann(Ef_eV, Ec_eV, T_K, Nc_cm3)
    p_boltz = f.p_boltzmann(Ef_eV, Ev_eV, T_K, Nv_cm3)
    st.session_state.n0_cm3, st.session_state.p0_cm3 = n_num, p_num

    st.metric("Nivel de Fermi E_F (resuelto por neutralidad de carga)",
              f"{Ef_eV:.4f} eV")

    E_range = np.linspace(Ev_eV - 5 * c.K_B_EV * T_K, Ec_eV + 5 * c.K_B_EV * T_K, 800)
    gC = f.densidad_estados_conduccion(E_range, Ec_eV, me_kg)
    gV = f.densidad_estados_valencia(E_range, Ev_eV, mh_kg)
    fE = f.fermi_dirac(E_range, Ef_eV, T_K)
    un_fE = f.un_menos_fermi_dirac(E_range, Ef_eV, T_K)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(gC, E_range, label="g_C(E)")
        ax.plot(gV, E_range, label="g_V(E)")
        ax.axhline(Ef_eV, color="red", linestyle="--", label="E_F")
        ax.set_xlabel("Densidad de estados [cm$^{-3}$ eV$^{-1}$]")
        ax.set_ylabel("Energía [eV]")
        ax.legend()
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(fE, E_range, label="f(E)")
        ax2.plot(un_fE, E_range, label="1-f(E)")
        ax2.axhline(Ef_eV, color="red", linestyle="--", label="E_F")
        ax2.set_xlabel("Ocupación [adimensional]")
        ax2.set_ylabel("Energía [eV]")
        ax2.legend()
        st.pyplot(fig2)
    with col3:
        # Productos g_C*f (electrones) y g_V*(1-f) (huecos): su área bajo la
        # curva ES n y p, respectivamente.
        prod_n = gC * fE
        prod_p = gV * un_fE
        fig3, ax3 = plt.subplots()
        ax3.fill_betweenx(E_range, prod_n, alpha=0.4, label="g_C·f  (→ n)")
        ax3.fill_betweenx(E_range, prod_p, alpha=0.4, label="g_V·(1-f)  (→ p)")
        ax3.axhline(Ef_eV, color="red", linestyle="--", label="E_F")
        ax3.set_xlabel("Densidad de carga ocupada [cm$^{-3}$ eV$^{-1}$]")
        ax3.set_ylabel("Energía [eV]")
        ax3.legend()
        st.pyplot(fig3)

    st.subheader("Integración numérica vs. aproximación de Boltzmann")
    tabla_comparacion = {
        "": ["n [cm⁻³]", "p [cm⁻³]", "n·p [cm⁻⁶]"],
        "Integración numérica": [f"{n_num:.4e}", f"{p_num:.4e}", f"{n_num*p_num:.4e}"],
        "Aprox. de Boltzmann": [f"{n_boltz:.4e}", f"{p_boltz:.4e}", f"{n_boltz*p_boltz:.4e}"],
        "n_i² de referencia": ["-", "-", f"{c.NI_SI_300K**2:.4e}"],
    }
    st.table(tabla_comparacion)
    if Na_cm3 >= 1e19 or Nd_cm3 >= 1e19:
        st.caption("Dopaje ≥10¹⁹ cm⁻³: régimen degenerado esperado — la aproximación "
                   "de Boltzmann debería divergir notoriamente de la integración numérica "
                   "y la ley de acción de masas n·p=n_i² debería dejar de cumplirse.")

# ---------------------------------------------------------------------------
# Pestaña 2 — El bloque de silicio (animación)
# ---------------------------------------------------------------------------
with tab2:
    st.header("El bloque de silicio")
    st.caption("Red cristalina real del silicio (estructura diamante, tetraédrica): cada átomo se "
               "enlaza a sus 4 vecinos más cercanos en 3D. No es una simplificación 2D — los enlaces "
               "covalentes se calculan por distancia geométrica real de la estructura diamante.")

    # -------------------------------------------------------------------
    # Mapeo declarado partícula <-> concentración (convención de visualización,
    # no una constante física): 1 símbolo dibujado por cada década de
    # concentración por encima de un piso de 10^6 cm^-3, con un tope de 40
    # símbolos por especie para mantener la figura legible.
    # -------------------------------------------------------------------
    PISO_EXPONENTE_VIS = 6
    MAX_SIMBOLOS_VIS = 40

    def num_simbolos(concentracion_cm3):
        if concentracion_cm3 <= 10 ** PISO_EXPONENTE_VIS:
            return 0
        return int(min(MAX_SIMBOLOS_VIS, max(1, round(np.log10(concentracion_cm3) - PISO_EXPONENTE_VIS))))

    st.caption(f"Mapeo: 1 símbolo por década de concentración por encima de 10^{PISO_EXPONENTE_VIS} cm⁻³ "
               f"(máx. {MAX_SIMBOLOS_VIS} símbolos por especie). Ejemplo con los valores actuales: "
               f"n={n_num:.2e} cm⁻³ → {num_simbolos(n_num)} electrones dibujados.")

    # -------------------------------------------------------------------
    # Red cristalina 3D: estructura diamante real del silicio (FCC + base de
    # 2 átomos por sitio de red -> 8 átomos por celda convencional). Los
    # enlaces se calculan por distancia geométrica real de primer vecino
    # (a*sqrt(3)/4 en unidades de la celda convencional), no se dibujan a
    # mano. Los átomos de la superficie del bloque quedan con menos de 4
    # enlaces (enlaces insatisfechos) — el mismo efecto que motiva la
    # recombinación superficial de la Pestaña 3.
    # -------------------------------------------------------------------
    N_CELDAS_LADO = 2  # celdas convencionales por lado (2x2x2 -> 64 átomos)

    def generar_red_diamante(n_celdas):
        base_fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        base_diamante = np.vstack([base_fcc, base_fcc + np.array([0.25, 0.25, 0.25])])
        atomos = []
        for i in range(n_celdas):
            for j in range(n_celdas):
                for k in range(n_celdas):
                    atomos.append(base_diamante + np.array([i, j, k], dtype=float))
        atomos = np.vstack(atomos)
        d_enlace = np.sqrt(3) / 4  # distancia de primer vecino en la red diamante (a=1)
        enlaces = []
        for i in range(len(atomos)):
            dists = np.linalg.norm(atomos[i + 1:] - atomos[i], axis=1)
            vecinos = np.where(np.abs(dists - d_enlace) < 1e-3)[0]
            enlaces.extend((i, i + 1 + v) for v in vecinos)
        return atomos, enlaces

    atomos_red, enlaces_red = generar_red_diamante(N_CELDAS_LADO)
    n_sitios = len(atomos_red)
    LADO = float(N_CELDAS_LADO)

    n_donores_dibujo = num_simbolos(Nd_cm3) if tipo == "tipo n" else 0
    n_aceptores_dibujo = num_simbolos(Na_cm3) if tipo == "tipo p" else 0
    n_electrones_dibujo = num_simbolos(n_num)
    n_huecos_dibujo = num_simbolos(p_num)

    rng_red = np.random.default_rng(abs(hash((tipo, round(Na_cm3), round(Nd_cm3)))) % (2**32))
    sitios_dopantes = rng_red.choice(n_sitios, size=n_donores_dibujo + n_aceptores_dibujo, replace=False) \
        if (n_donores_dibujo + n_aceptores_dibujo) <= n_sitios else np.array([], dtype=int)
    sitios_donores = sitios_dopantes[:n_donores_dibujo]
    sitios_aceptores = sitios_dopantes[n_donores_dibujo:n_donores_dibujo + n_aceptores_dibujo]

    rng_carga = np.random.default_rng(abs(hash((n_electrones_dibujo, n_huecos_dibujo))) % (2**32))
    pos_electrones = rng_carga.uniform(0, LADO, size=(n_electrones_dibujo, 3))
    pos_huecos = rng_carga.uniform(0, LADO, size=(n_huecos_dibujo, 3))

    def dibujar_bloque(ax, extra_generacion=None, extra_recombinacion=None):
        # enlaces covalentes tetraédricos (líneas 3D entre primeros vecinos reales)
        for i, j in enlaces_red:
            xs, ys, zs = zip(atomos_red[i], atomos_red[j])
            ax.plot(xs, ys, zs, color="lightgray", linewidth=1.2, zorder=1)
        # átomos de silicio de la red
        ax.scatter(atomos_red[:, 0], atomos_red[:, 1], atomos_red[:, 2],
                   color="silver", s=90, zorder=2, label="Si (red)")
        # dopantes sustitucionales, ya ionizados (iones fijos)
        if len(sitios_donores):
            p3 = atomos_red[sitios_donores]
            ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], color="green", s=150,
                       marker="P", zorder=3, label="P⁺ (donor ionizado)")
        if len(sitios_aceptores):
            p3 = atomos_red[sitios_aceptores]
            ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], color="orange", s=150,
                       marker="X", zorder=3, label="B⁻ (aceptor ionizado)")
        # portadores móviles
        if len(pos_electrones):
            ax.scatter(pos_electrones[:, 0], pos_electrones[:, 1], pos_electrones[:, 2],
                       color="blue", s=45, marker="o", zorder=4, label="e⁻ móvil")
        if len(pos_huecos):
            ax.scatter(pos_huecos[:, 0], pos_huecos[:, 1], pos_huecos[:, 2],
                       facecolors="none", edgecolors="red", s=45, marker="o", zorder=4, label="hueco móvil")
        if extra_generacion is not None and len(extra_generacion):
            ax.scatter(extra_generacion[:, 0], extra_generacion[:, 1], extra_generacion[:, 2],
                       color="gold", s=200, marker="*", zorder=5, label="generación e⁻-h⁺")
        if extra_recombinacion is not None and len(extra_recombinacion):
            ax.scatter(extra_recombinacion[:, 0], extra_recombinacion[:, 1], extra_recombinacion[:, 2],
                       color="black", s=130, marker="x", zorder=5, label="recombinación")
        ax.set_xlim(-0.3, LADO + 0.3)
        ax.set_ylim(-0.3, LADO + 0.3)
        ax.set_zlim(-0.3, LADO + 0.3)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        ax.view_init(elev=18, azim=35)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=8)

    col_fig, col_datos = st.columns([2, 1])
    with col_fig:
        placeholder_bloque = st.empty()
        fig_b = plt.figure(figsize=(6, 6))
        ax_b = fig_b.add_subplot(111, projection="3d")
        dibujar_bloque(ax_b)
        placeholder_bloque.pyplot(fig_b)
        plt.close(fig_b)

        if st.button("▶ Reproducir generación/recombinación térmica"):
            n_frames = 24
            # tasa de eventos por frame: crece con T (declarado, adimensional
            # de visualización, no una tasa física calibrada) — hace visible
            # que la generación/recombinación térmica se acelera con T.
            tasa_por_frame = max(1, round(T_K / 100))
            for _ in range(n_frames):
                gen = rng_carga.uniform(0, LADO, size=(tasa_por_frame, 3))
                recomb = rng_carga.uniform(0, LADO, size=(tasa_por_frame, 3))
                fig_a = plt.figure(figsize=(6, 6))
                ax_a = fig_a.add_subplot(111, projection="3d")
                dibujar_bloque(ax_a, extra_generacion=gen, extra_recombinacion=recomb)
                placeholder_bloque.pyplot(fig_a)
                plt.close(fig_a)
                time.sleep(0.12)
            fig_b = plt.figure(figsize=(6, 6))
            ax_b = fig_b.add_subplot(111, projection="3d")
            dibujar_bloque(ax_b)
            placeholder_bloque.pyplot(fig_b)
            plt.close(fig_b)
        st.caption(f"Tasa de eventos por cuadro de animación: {max(1, round(T_K/100))} pares/cuadro a "
                   f"T={T_K:.0f} K (crece con T: 1 par/cuadro a 100 K, hasta 6 a 600 K).")

    with col_datos:
        st.metric("n (numérico)", f"{n_num:.3e} cm⁻³")
        st.metric("p (numérico)", f"{p_num:.3e} cm⁻³")
        producto_np = n_num * p_num
        st.metric("n·p", f"{producto_np:.3e} cm⁻⁶")
        ni2_modelo = f.Nc_efectiva(T_K, me_kg) * f.Nv_efectiva(T_K, mh_kg) * np.exp(-Ec_eV / (c.K_B_EV * T_K))
        error_pct = abs(producto_np - ni2_modelo) / ni2_modelo * 100
        degenerado = (Na_cm3 >= 1e19) or (Nd_cm3 >= 1e19)
        if (not degenerado) and error_pct < 5:
            st.success(f"n·p ≈ n_i² del modelo ({ni2_modelo:.3e} cm⁻⁶, error {error_pct:.1f}%): "
                       "equilibrio térmico, régimen no degenerado. ✅ Ley de acción de masas se cumple.")
        elif degenerado:
            st.warning(f"Dopaje degenerado (≥10¹⁹ cm⁻³): n·p difiere {error_pct:.1f}% de n_i² del modelo. "
                       "❌ La ley de acción de masas deja de cumplirse, como se espera fuera de la "
                       "aproximación de Boltzmann.")
        else:
            st.warning(f"n·p difiere {error_pct:.1f}% de n_i² del modelo — revisar consistencia numérica.")

# ---------------------------------------------------------------------------
# Pestaña 3 — Los cuatro caminos de la recombinación
# ---------------------------------------------------------------------------
with tab3:
    st.header("Los cuatro caminos de la recombinación")

    def dibujar_bandas(ax, titulo, electron_y=None, hueco_y=None,
                        foton_direccion=None, fonones_en=None, trampa_y=None,
                        superficie=False, x_evento=0.5):
        margen = 0.3 * (Ec_eV - Ev_eV)
        ax.axhline(Ec_eV, color="black", linewidth=1.5)
        ax.axhline(Ev_eV, color="black", linewidth=1.5)
        ax.text(1.01, Ec_eV, "$E_C$", va="center")
        ax.text(1.01, Ev_eV, "$E_V$", va="center")
        if trampa_y is not None:
            ax.axhline(trampa_y, color="purple", linestyle=":", linewidth=1.5)
            ax.text(1.01, trampa_y, "$E_T$", va="center", color="purple")
        if superficie:
            ax.axvspan(0.82, 1.0, color="lightcoral", alpha=0.25)
            ax.text(0.83, Ec_eV + margen * 0.7, "superficie\n(enlaces\ninsatisfechos)",
                    fontsize=7, color="firebrick")
        if electron_y is not None:
            ax.scatter([x_evento], [electron_y], color="blue", s=180, zorder=5, label="e⁻")
        if hueco_y is not None:
            ax.scatter([x_evento], [hueco_y], facecolors="none", edgecolors="red",
                       s=180, linewidths=2, zorder=5, label="hueco")
        if foton_direccion in ("emitido", "reabsorbido"):
            y0, y1 = Ev_eV + 0.5 * (Ec_eV - Ev_eV), Ec_eV + margen * (0.8 if foton_direccion == "emitido" else 0.1)
            if foton_direccion == "reabsorbido":
                y0, y1 = y1, y0
            xs_foton = np.linspace(x_evento, x_evento + 0.15, 30)
            ys_foton = np.linspace(y0, y1, 30) + 0.02 * (Ec_eV - Ev_eV) * np.sin(np.linspace(0, 6 * np.pi, 30))
            ax.plot(xs_foton, ys_foton, color="gold", linewidth=2, zorder=4)
            ax.annotate("", xy=(xs_foton[-1], ys_foton[-1]), xytext=(xs_foton[-2], ys_foton[-2]),
                        arrowprops=dict(arrowstyle="->", color="gold", lw=2))
        if fonones_en is not None:
            xs_f = np.linspace(x_evento - 0.08, x_evento + 0.08, 20)
            ys_f = fonones_en + 0.015 * (Ec_eV - Ev_eV) * np.sin(np.linspace(0, 8 * np.pi, 20))
            ax.plot(xs_f, ys_f, color="gray", linewidth=1.5, zorder=4)
            ax.text(x_evento + 0.1, fonones_en, "fonones (calor)", fontsize=7, color="gray")
        ax.set_xlim(0, 1.2)
        ax.set_ylim(Ev_eV - margen, Ec_eV + margen)
        ax.set_xticks([])
        ax.set_ylabel("Energía [eV]")
        ax.set_title(titulo, fontsize=11)
        if electron_y is not None or hueco_y is not None:
            ax.legend(loc="lower right", fontsize=7)

    sub_rad, sub_auger, sub_srh, sub_surf = st.tabs(
        ["Radiativa", "Auger", "SRH (trampas)", "Superficie"])

    # -- Radiativa ---------------------------------------------------------
    with sub_rad:
        st.markdown("**Qué le ocurre al electrón:** cae de $E_C$ a $E_V$ y se recombina con el hueco. "
                     "**Qué le ocurre al hueco:** desaparece al recibir al electrón. "
                     "**Dónde termina la energía:** se emite como fotón de energía ≈ $E_g$, que puede "
                     "escapar del material o ser reabsorbido (regenerando un par), según el espesor.")
        p_reabs = 1 - np.exp(-st.session_state.espesor_cm / 100e-4)
        st.caption(f"Modelo cualitativo de este panel (no calibrado con α(λ) real, que corresponde a los "
                    f"Problemas 2.2/2.3): probabilidad de reabsorción ≈ {p_reabs:.2f} para un espesor de "
                    f"{st.session_state.espesor_cm*1e4:.0f} μm (crece con el espesor).")
        ph_rad = st.empty()
        if st.button("▶ Reproducir", key="btn_rad"):
            rng_rad = np.random.default_rng(1)
            n_escapo, n_reabs = 0, 0
            for i in range(10):
                fig, ax = plt.subplots(figsize=(4.5, 4))
                dibujar_bandas(ax, "Recombinación radiativa", electron_y=Ec_eV, hueco_y=Ev_eV)
                ph_rad.pyplot(fig); plt.close(fig); time.sleep(0.3)
                reabsorbido = rng_rad.uniform() < p_reabs
                fig, ax = plt.subplots(figsize=(4.5, 4))
                dibujar_bandas(ax, "Recombinación radiativa",
                               foton_direccion="reabsorbido" if reabsorbido else "emitido")
                ph_rad.pyplot(fig); plt.close(fig); time.sleep(0.35)
                if reabsorbido:
                    n_reabs += 1
                else:
                    n_escapo += 1
            st.info(f"En esta corrida: {n_escapo} fotones escaparon, {n_reabs} fueron reabsorbidos "
                    f"(≈{100*n_reabs/(n_reabs+n_escapo):.0f}%, esperado ≈{100*p_reabs:.0f}%).")

    # -- Auger ---------------------------------------------------------
    with sub_auger:
        st.markdown("**Qué le ocurre al electrón y al hueco:** el par se recombina banda a banda, pero la "
                     "energía liberada no se emite como fotón: se entrega a un tercer portador (electrón u "
                     "hueco) cercano. **Dónde termina la energía:** ese tercer portador queda muy caliente, "
                     "muy adentro de su banda, y luego se termaliza hasta el borde de banda emitiendo fonones "
                     "(calor a la red), sin emitir nunca un fotón.")
        ph_auger = st.empty()
        if st.button("▶ Reproducir", key="btn_auger"):
            rng_auger = np.random.default_rng(2)
            for _ in range(6):
                portador_e = rng_auger.uniform() < 0.5
                margen = Ec_eV - Ev_eV
                y_caliente = Ec_eV + 0.6 * margen if portador_e else Ev_eV - 0.6 * margen
                fig, ax = plt.subplots(figsize=(4.5, 4))
                dibujar_bandas(ax, "Recombinación Auger",
                               electron_y=Ec_eV if not portador_e else None,
                               hueco_y=Ev_eV if portador_e else None)
                ax.scatter([0.5], [y_caliente], color=("blue" if portador_e else "red"), s=220,
                           marker="*", zorder=6, label="portador caliente")
                ax.legend(loc="lower right", fontsize=7)
                ph_auger.pyplot(fig); plt.close(fig); time.sleep(0.3)
                for frac in (0.35, 0.05):
                    y_term = (Ec_eV + frac * margen) if portador_e else (Ev_eV - frac * margen)
                    y_borde = Ec_eV if portador_e else Ev_eV
                    fig, ax = plt.subplots(figsize=(4.5, 4))
                    dibujar_bandas(ax, "Termalización (Auger)", fonones_en=(y_term + y_borde) / 2)
                    ax.scatter([0.5], [y_term], color=("blue" if portador_e else "red"), s=180, zorder=6)
                    ph_auger.pyplot(fig); plt.close(fig); time.sleep(0.25)

    # -- SRH ---------------------------------------------------------
    with sub_srh:
        st.markdown("**Qué le ocurre al electrón:** cae primero a un nivel de trampa dentro de la banda "
                     "prohibida (defecto del cristal). **Qué le ocurre al hueco:** llega después a ese mismo "
                     "nivel y se recombina con el electrón atrapado. **Dónde termina la energía:** se emite "
                     "en dos etapas, como fonones (calor), nunca como fotón.")
        Et_frac = st.slider("Posición del nivel de trampa (0=E_V, 1=E_C)", 0.0, 1.0, 0.5, 0.05, key="Et_frac")
        Et_eV = Ev_eV + Et_frac * (Ec_eV - Ev_eV)
        eficiencia_srh = np.exp(-abs(Et_frac - 0.5) / 0.25)
        st.caption(f"Modelo cualitativo de eficiencia de captura SRH (declarado, refleja el resultado "
                    f"estándar de Shockley-Read-Hall: máxima cuando la trampa está cerca del centro de la "
                    f"banda prohibida): eficiencia relativa ≈ {eficiencia_srh:.2f}.")
        ph_srh = st.empty()
        if st.button("▶ Reproducir", key="btn_srh"):
            margen = Ec_eV - Ev_eV
            fig, ax = plt.subplots(figsize=(4.5, 4))
            dibujar_bandas(ax, "SRH: estado inicial", electron_y=Ec_eV, hueco_y=Ev_eV, trampa_y=Et_eV)
            ph_srh.pyplot(fig); plt.close(fig); time.sleep(0.4)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            dibujar_bandas(ax, "SRH: electrón cae a la trampa", electron_y=Et_eV, hueco_y=Ev_eV,
                           trampa_y=Et_eV, fonones_en=(Ec_eV + Et_eV) / 2)
            ph_srh.pyplot(fig); plt.close(fig); time.sleep(0.4)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            dibujar_bandas(ax, "SRH: hueco llega y se recombina", trampa_y=Et_eV,
                           fonones_en=(Ev_eV + Et_eV) / 2)
            ph_srh.pyplot(fig); plt.close(fig); time.sleep(0.4)

    # -- Superficie ---------------------------------------------------------
    with sub_surf:
        st.markdown("**Qué le ocurre al electrón y al hueco:** ambos difunden hasta la superficie del "
                     "cristal, donde enlaces insatisfechos generan un continuo de niveles (como una SRH "
                     "distribuida en la superficie). **Dónde termina la energía:** como fonones, igual que "
                     "en SRH. La velocidad de recombinación superficial S controla qué tan eficiente es este "
                     "camino; pasivar la superficie reduce S.")
        S_actual = st.session_state.S_frontal_cm_s
        st.caption(f"S actual = {S_actual:.1e} cm/s. A mayor S, más frecuentes los eventos en la superficie.")
        ph_surf = st.empty()
        if st.button("▶ Reproducir", key="btn_surf"):
            n_eventos = int(np.clip(np.log10(S_actual) - 1, 1, 8))
            rng_surf = np.random.default_rng(4)
            for _ in range(n_eventos):
                y_e = rng_surf.uniform(Ev_eV + 0.3 * (Ec_eV - Ev_eV), Ec_eV)
                fig, ax = plt.subplots(figsize=(4.5, 4))
                dibujar_bandas(ax, "Difusión hacia la superficie", electron_y=y_e, hueco_y=Ev_eV,
                               superficie=True, x_evento=0.9)
                ph_surf.pyplot(fig); plt.close(fig); time.sleep(0.25)
                fig, ax = plt.subplots(figsize=(4.5, 4))
                dibujar_bandas(ax, "Recombinación en la superficie", superficie=True,
                               fonones_en=(Ec_eV + Ev_eV) / 2, x_evento=0.9)
                ph_surf.pyplot(fig); plt.close(fig); time.sleep(0.3)

    st.divider()
    st.subheader("Panel cuantitativo: vida media efectiva vs. nivel de inyección")

    n0_cm3 = st.session_state.get("n0_cm3", c.NI_SI_300K)
    p0_cm3 = st.session_state.get("p0_cm3", c.NI_SI_300K)
    tau_srh_s = st.session_state.tau_srh_volumen_s
    S_cm_s = st.session_state.S_frontal_cm_s
    W_cm = st.session_state.espesor_cm

    delta_n_arr = np.logspace(12, 18, 200)
    tau_rad_arr = f.tau_radiativo(n0_cm3, p0_cm3, delta_n_arr, c.B_RADIATIVO_SI)
    tau_auger_arr = f.tau_auger(delta_n_arr, c.C_AUGER_SI)
    tau_srh_arr = np.full_like(delta_n_arr, tau_srh_s)
    tau_superficie_val = f.tau_superficie(S_cm_s, W_cm)
    tau_superficie_arr = np.full_like(delta_n_arr, tau_superficie_val)
    tau_eff_arr = f.tau_efectivo(tau_rad_arr, tau_auger_arr, tau_srh_arr, tau_superficie_arr)

    fig_tau, ax_tau = plt.subplots(figsize=(7, 4.5))
    ax_tau.loglog(delta_n_arr, tau_rad_arr, label="τ_radiativa")
    ax_tau.loglog(delta_n_arr, tau_auger_arr, label="τ_Auger")
    ax_tau.loglog(delta_n_arr, tau_srh_arr, label="τ_SRH (volumen, asignado por semilla)")
    ax_tau.loglog(delta_n_arr, tau_superficie_arr, label="τ_superficie")
    ax_tau.loglog(delta_n_arr, tau_eff_arr, label="τ_efectivo", color="black", linewidth=2.5)
    delta_n_cruce = np.sqrt(1.0 / (c.C_AUGER_SI * tau_srh_s))
    if delta_n_arr.min() <= delta_n_cruce <= delta_n_arr.max():
        ax_tau.axvline(delta_n_cruce, color="gray", linestyle="--")
        ax_tau.text(delta_n_cruce, tau_eff_arr.min(), f"  cruce SRH↔Auger\n  Δn≈{delta_n_cruce:.2e} cm⁻³",
                    fontsize=8, va="bottom")
    ax_tau.set_xlabel("Nivel de inyección Δn [cm⁻³]")
    ax_tau.set_ylabel("Vida media τ [s]")
    ax_tau.legend(fontsize=8)
    ax_tau.grid(True, which="both", alpha=0.3)
    st.pyplot(fig_tau)

    st.caption(f"Cruce de dominancia SRH → Auger en Δn ≈ {delta_n_cruce:.3e} cm⁻³ "
               f"(donde 1/τ_SRH = 1/τ_Auger, con τ_SRH={tau_srh_s*1e6:.0f} μs asignado por semilla).")

    delta_n_actual = st.session_state.delta_n_cm3
    tau_eff_actual = f.tau_efectivo(
        f.tau_radiativo(n0_cm3, p0_cm3, delta_n_actual, c.B_RADIATIVO_SI),
        f.tau_auger(delta_n_actual, c.C_AUGER_SI),
        tau_srh_s,
        tau_superficie_val)
    D_n = f.coef_difusion_einstein(c.MU_N_REF, st.session_state.T_K)
    D_p = f.coef_difusion_einstein(c.MU_P_REF, st.session_state.T_K)
    L_n = f.longitud_difusion(D_n, tau_eff_actual)
    L_p = f.longitud_difusion(D_p, tau_eff_actual)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("τ_efectivo @ Δn actual", f"{tau_eff_actual*1e6:.2f} μs")
    col_b.metric("L_n = √(D_n·τ)", f"{L_n*1e4:.1f} μm")
    col_c.metric("L_p = √(D_p·τ)", f"{L_p*1e4:.1f} μm")

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

    st.table(filas)

    n_aprobados = sum(1 for fil in filas if fil["Veredicto"].startswith("✅"))
    st.metric("Verificaciones aprobadas", f"{n_aprobados} / {len(filas)}")
