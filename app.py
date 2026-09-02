from __future__ import annotations

import math
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ======================================================================
# Constants - preserved from the supplied desktop laboratory
# ======================================================================
HBAR = 1.055
MASS = 9.1 * 1.6
TWO_M_OVER_HBAR2 = 2.0 * MASS / HBAR ** 2

X_MIN, X_MAX = -3.0, 3.0
N_POINTS = 4000
BUFFER = 0.20

E_MIN, E_MAX, E_STEP = 0.001, 2.000, 0.001
V_MIN, V_MAX, V_STEP = 0.001, 2.000, 0.001
A_MIN, A_MAX, A_STEP = 0.010, 1.000, 0.010

INSTRUCTOR = os.environ.get("TUNNELING_INSTRUCTOR", "0") == "1"

# Appearance values preserved conceptually from the desktop lab.
GRID_MAJOR_COLOUR = "#3d6ea8"
GRID_MINOR_COLOUR = "#89a8cc"
CURVE_WIDTH_MAIN = 1.6
CURVE_WIDTH_MEASURE = 2.0
TRANSMITTED_COLOUR = "red"
INSTRUCTION = (
    "Measure |t| from the lower graph, then the transmission coefficient "
    "is  T = |t|²"
)


# ======================================================================
# Physics - same equations and 4x4 continuity system
# ======================================================================
def wave_numbers(E, V0):
    if abs(V0 - E) < 1.0e-9:
        V0 = E + 1.0e-9

    k = math.sqrt(TWO_M_OVER_HBAR2 * E)
    kappa = np.sqrt(complex(TWO_M_OVER_HBAR2 * (V0 - E)))
    return k, kappa


def solve_coefficients(E, V0, a):
    k, kappa = wave_numbers(E, V0)

    ep, em = np.exp(kappa * a), np.exp(-kappa * a)
    eik = np.exp(1j * k * a)

    M = np.array([
        [-1.0,     1.0,        1.0,         0.0],
        [-1j * k, -kappa,      kappa,       0.0],
        [0.0,      ep,         em,         -eik],
        [0.0,      kappa * ep, -kappa * em, -1j * k * eik],
    ], dtype=complex)

    rhs = np.array([1.0, -1j * k, 0.0, 0.0], dtype=complex)
    r, C, D, t = np.linalg.solve(M, rhs)
    return r, C, D, t, k, kappa


def wavefunction(x, r, C, D, t, k, kappa, a):
    psi = np.zeros_like(x, dtype=complex)

    i1 = x < 0.0
    i2 = (x >= 0.0) & (x <= a)
    i3 = x > a

    psi[i1] = np.exp(1j * k * x[i1]) + r * np.exp(-1j * k * x[i1])
    psi[i2] = C * np.exp(kappa * x[i2]) + D * np.exp(-kappa * x[i2])
    psi[i3] = t * np.exp(1j * k * x[i3])

    return psi


def transmission_analytical(E, V0, a):
    if abs(V0 - E) < 1.0e-9:
        V0 = E + 1.0e-9

    if E < V0:
        kap = math.sqrt(TWO_M_OVER_HBAR2 * (V0 - E))
        return 1.0 / (
            1.0
            + V0 ** 2 * math.sinh(kap * a) ** 2
            / (4.0 * E * (V0 - E))
        )

    q = math.sqrt(TWO_M_OVER_HBAR2 * (E - V0))
    return 1.0 / (
        1.0
        + V0 ** 2 * math.sin(q * a) ** 2
        / (4.0 * E * (E - V0))
    )


_NICE = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def nice_limit(value):
    if value <= 0 or not np.isfinite(value):
        return 1.0

    exponent = math.floor(math.log10(value))
    frac = value / 10.0 ** exponent

    for m in _NICE:
        if frac <= m + 1e-12:
            return m * 10.0 ** exponent

    return 10.0 ** (exponent + 1)


def format_div(value):
    return f"{value:.4g}" if value >= 0.01 else f"{value:.1e}"


# ======================================================================
# Plotly graph helpers
# ======================================================================
def apply_common_layout(fig, height=620):
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=85, t=90, b=55),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_MAJOR_COLOUR,
        gridwidth=1,
        minor=dict(showgrid=True, gridcolor=GRID_MINOR_COLOUR, gridwidth=0.5),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_MAJOR_COLOUR,
        gridwidth=1,
        minor=dict(showgrid=True, gridcolor=GRID_MINOR_COLOUR, gridwidth=0.5),
    )
    return fig


def make_tunneling_figure(E, V0, a, x, psi, t_abs, x0):
    V = np.where((x >= 0.0) & (x <= a), V0, 0.0)

    y_top = nice_limit(1.05 * float(np.max(np.abs(psi))))
    v_top = nice_limit(1.35 * max(V0, E))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.20,
        specs=[[{"secondary_y": True}], [{}]],
        row_heights=[0.45, 0.55],
        subplot_titles=[
            "Quantum tunnelling through a rectangular barrier",
            "MEASUREMENT WINDOW — read the peak height of the red curve = |t|",
        ],
    )

    left = x <= a
    right = x >= a

    fig.add_trace(
        go.Scatter(
            x=x[left],
            y=np.real(psi[left]),
            mode="lines",
            name="Re ψ(x)",
            line=dict(width=CURVE_WIDTH_MAIN),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x[right],
            y=np.real(psi[right]),
            mode="lines",
            name="transmitted",
            line=dict(width=CURVE_WIDTH_MAIN, color=TRANSMITTED_COLOUR),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.abs(psi),
            mode="lines",
            name="|ψ(x)|",
            line=dict(width=0.9, dash="dash"),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # Barrier and measured region shading.
    fig.add_vrect(
        x0=0, x1=a,
        fillcolor="gray",
        opacity=0.18,
        line_width=0,
        row=1, col=1,
    )
    fig.add_vrect(
        x0=x0, x1=X_MAX,
        fillcolor="red",
        opacity=0.06,
        line_width=0,
        row=1, col=1,
    )

    # ±|t| measurement markers.
    fig.add_hline(
        y=t_abs, line_dash="dot", line_width=1.4,
        line_color=TRANSMITTED_COLOUR, row=1, col=1,
    )
    fig.add_hline(
        y=-t_abs, line_dash="dot", line_width=1.4,
        line_width=1.4,
        line_color=TRANSMITTED_COLOUR, row=1, col=1,
    )

    fig.add_annotation(
        x=X_MAX, y=t_abs, xanchor="right",
        text="|t|", showarrow=False,
        font=dict(color=TRANSMITTED_COLOUR),
        row=1, col=1,
    )

    # Energy/barrier axis on the right.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=V,
            mode="lines",
            name="V(x)",
            line=dict(width=1.6, color="magenta"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.full_like(x, E),
            mode="lines",
            name="E",
            line=dict(width=1.4, color="green", dash="dash"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Measurement panel: transmitted wave only.
    xm = np.linspace(x0, X_MAX, 2000)
    _, _, _, _, k, _ = solve_coefficients(E, V0, a)
    psi_m = solve_coefficients(E, V0, a)[3] * np.exp(
        1j * k * xm
    )

    y_bot = nice_limit(1.20 * t_abs)
    minor = (y_bot / 4.0) / 5.0

    fig.add_trace(
        go.Scatter(
            x=xm,
            y=np.real(psi_m),
            mode="lines",
            name="Measured transmitted wave",
            line=dict(width=CURVE_WIDTH_MEASURE, color=TRANSMITTED_COLOUR),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0, line_width=0.8, row=2, col=1
    )

    fig.update_xaxes(
        range=[X_MIN, X_MAX],
        title_text="x",
        row=1, col=1,
    )
    fig.update_yaxes(
        range=[-y_top, y_top],
        title_text="Amplitude (incident wave = 1)",
        row=1, col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        range=[-0.08 * v_top, v_top],
        title_text="Energy (V₀ and E)",
        row=1, col=1,
        secondary_y=True,
    )

    fig.update_xaxes(
        range=[x0, X_MAX],
        title_text="x   (region III, beyond the barrier)",
        row=2, col=1,
    )
    fig.update_yaxes(
        range=[-y_bot, y_bot],
        title_text="Transmitted amplitude<br>(same scale, magnified)",
        row=2, col=1,
    )

    fig.add_annotation(
        x=0.5, y=-0.19,
        xref="paper", yref="paper",
        text=f"1 small division = {format_div(minor)}",
        showarrow=False,
        font=dict(size=12),
    )

    apply_common_layout(fig, height=850)
    return fig


# ======================================================================
# Streamlit application
# ======================================================================
st.set_page_config(
    page_title="Quantum Tunnelling through a Rectangular Barrier",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Quantum Tunnelling through a Rectangular Barrier")
st.caption(
    "Faithful browser conversion of the supplied desktop laboratory. "
    "The physics and numerical method are preserved; only the desktop GUI "
    "has been replaced by Streamlit + Plotly."
)

with st.sidebar:
    st.header("Experiment controls")

    E = st.slider(
        "Particle energy E",
        min_value=float(E_MIN),
        max_value=float(E_MAX),
        value=0.300,
        step=float(E_STEP),
        format="%.3f",
    )

    V0 = st.slider(
        "Barrier height V₀",
        min_value=float(V_MIN),
        max_value=float(V_MAX),
        value=0.500,
        step=float(V_STEP),
        format="%.3f",
    )

    a = st.slider(
        "Barrier width a",
        min_value=float(A_MIN),
        max_value=float(A_MAX),
        value=0.500,
        step=float(A_STEP),
        format="%.3f",
    )

    st.divider()
    st.markdown(
        "**Measurement instruction**  \n"
        "Measure the peak height of the red transmitted wave in the "
        "lower graph. That height is |t|."
    )

r, C, D, t, k, kappa = solve_coefficients(E, V0, a)

x = np.linspace(X_MIN, X_MAX, N_POINTS)
psi = wavefunction(x, r, C, D, t, k, kappa, a)

t_abs = float(abs(t))
T = t_abs ** 2
R = float(abs(r)) ** 2

x0 = min(a + BUFFER, X_MAX - 0.5)

fig = make_tunneling_figure(E, V0, a, x, psi, t_abs, x0)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Readout")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("E", f"{E:.3f}")
with c2:
    st.metric("V₀", f"{V0:.3f}")
with c3:
    st.metric("a", f"{a:.3f}")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("|t|", f"{t_abs:.5f}")
with c2:
    st.metric("T = |t|²", f"{T:.5f}")
with c3:
    st.metric("R = |r|²", f"{R:.5f}")

st.info(INSTRUCTION)

st.code(
    f"|t| = {t_abs:.5f}    T = {T:.5f}    "
    f"R = {R:.5f}    R + T = {R + T:.6f}",
    language="text",
)

if INSTRUCTOR:
    st.warning(
        f"INSTRUCTOR KEY   |t| = {t_abs:.5f}   "
        f"T = {T:.5f}   R = {R:.5f}   R+T = {R + T:.6f}   "
        f"T(analytic) = {transmission_analytical(E, V0, a):.5f}"
    )

with st.expander("Physics / numerical details"):
    st.markdown(
        """
**Region I:**  ψ = e^(ikx) + r e^(-ikx)

**Region II:** ψ = C e^(κx) + D e^(-κx)

**Region III:** ψ = t e^(ikx)

The four continuity conditions at x = 0 and x = a are solved as a
4×4 complex linear system. This same code path handles both E < V₀
and E > V₀.

The incident amplitude is fixed at 1, so the transmitted amplitude is
|t| and the transmission coefficient is T = |t|².
"""
    )

st.caption(
    "No Qiskit, Jupyter, Tkinter, Qt, or desktop Matplotlib backend is required."
)
