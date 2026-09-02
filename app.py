import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantum Tunnelling Laboratory",
    page_icon="⚛️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

HBAR = 1.055
MASS = 9.1 * 1.6

TWO_M_OVER_HBAR2 = 2.0 * MASS / HBAR**2

X_MIN = -3.0
X_MAX = 3.0
N_POINTS = 4000

BUFFER = 0.20

E_MIN = 0.001
E_MAX = 2.000
E_STEP = 0.001
E_DEFAULT = 0.300

V0_MIN = 0.001
V0_MAX = 2.000
V0_STEP = 0.001
V0_DEFAULT = 0.500

A_MIN = 0.010
A_MAX = 1.000
A_STEP = 0.010
A_DEFAULT = 0.500


# ============================================================
# DISPLAY SETTINGS
# ============================================================

TRANSMITTED_COLOUR = "red"
BARRIER_COLOUR = "rgba(100,100,100,0.25)"
MEASURED_COLOUR = "rgba(255,165,0,0.15)"
GRID_COLOUR = "rgba(120,120,120,0.25)"

INSTRUCTOR_KEY = os.environ.get("TUNNELING_INSTRUCTOR", "0") == "1"


# ============================================================
# PAGE TITLE
# ============================================================

st.title("⚛️ Quantum Tunnelling through a Rectangular Barrier")

st.markdown(
    """
This laboratory simulates a quantum particle incident on a rectangular
potential barrier.

For a particle with energy **E < V₀**, the wavefunction penetrates the
barrier and a transmitted wave appears beyond it.
"""
)


# ============================================================
# PHYSICS FUNCTIONS
# ============================================================

def wave_numbers(E, V0):
    """
    Calculate the wave number k in the free-particle regions and
    the decay constant kappa inside the barrier.
    """

    k = np.sqrt(TWO_M_OVER_HBAR2 * E)

    if E < V0:
        kappa = np.sqrt(TWO_M_OVER_HBAR2 * (V0 - E))
    else:
        # For E >= V0, use an imaginary kappa so that the same
        # mathematical formulation remains usable.
        kappa = 1j * np.sqrt(TWO_M_OVER_HBAR2 * (E - V0))

    return k, kappa


def solve_coefficients(E, V0, a):
    """
    Solve the four continuity equations for:

        Region I:
            psi = exp(ikx) + r exp(-ikx)

        Region II:
            psi = C exp(kappa x) + D exp(-kappa x)

        Region III:
            psi = t exp(ikx)

    Unknowns:
        r, C, D, t
    """

    k, kappa = wave_numbers(E, V0)

    # --------------------------------------------------------
    # Boundary at x = 0
    #
    # 1 + r = C + D
    # ik(1-r) = kappa(C-D)
    # --------------------------------------------------------

    row1 = [
        1,
        -1,
        -1,
        0,
    ]

    row2 = [
        -1j * k,
        -kappa,
        kappa,
        0,
    ]

    # --------------------------------------------------------
    # Boundary at x = a
    #
    # C exp(kappa a) + D exp(-kappa a)
    #     = t exp(ika)
    #
    # kappa[C exp(kappa a) - D exp(-kappa a)]
    #     = ik t exp(ika)
    # --------------------------------------------------------

    exp_kappa_a = np.exp(kappa * a)
    exp_minus_kappa_a = np.exp(-kappa * a)
    exp_ika = np.exp(1j * k * a)

    row3 = [
        0,
        exp_kappa_a,
        exp_minus_kappa_a,
        -exp_ika,
    ]

    row4 = [
        0,
        kappa * exp_kappa_a,
        -kappa * exp_minus_kappa_a,
        -1j * k * exp_ika,
    ]

    matrix = np.array(
        [
            row1,
            row2,
            row3,
            row4,
        ],
        dtype=complex,
    )

    # Right-hand side
    rhs = np.array(
        [
            -1,
            -1j * k,
            0,
            0,
        ],
        dtype=complex,
    )

    r, C, D, t = np.linalg.solve(matrix, rhs)

    return r, C, D, t, k, kappa


def wavefunction(x, r, C, D, t, k, kappa, a):
    """
    Calculate the wavefunction in all three spatial regions.

    Region I:
        x < 0

    Region II:
        0 <= x <= a

    Region III:
        x > a
    """

    x = np.asarray(x)

    psi = np.zeros_like(x, dtype=complex)

    region_1 = x < 0
    region_2 = (x >= 0) & (x <= a)
    region_3 = x > a

    # Region I
    psi[region_1] = (
        np.exp(1j * k * x[region_1])
        + r * np.exp(-1j * k * x[region_1])
    )

    # Region II
    psi[region_2] = (
        C * np.exp(kappa * x[region_2])
        + D * np.exp(-kappa * x[region_2])
    )

    # Region III
    psi[region_3] = (
        t * np.exp(1j * k * x[region_3])
    )

    return psi


def transmission_analytical(E, V0, a):
    """
    Analytical transmission coefficient for a rectangular barrier.

    For E < V0:

        T = 1 /
            [1 + (V0^2 sinh^2(kappa a))
             / (4 E (V0-E))]

    For E >= V0:

        T = 1 /
            [1 + (V0^2 sin^2(k a))
             / (4 E (E-V0))]
    """

    if E < V0:

        kappa = np.sqrt(
            TWO_M_OVER_HBAR2 * (V0 - E)
        )

        denominator = (
            1
            + (
                V0**2
                * np.sinh(kappa * a) ** 2
            )
            / (
                4 * E * (V0 - E)
            )
        )

        return 1.0 / denominator

    elif E > V0:

        q = np.sqrt(
            TWO_M_OVER_HBAR2 * (E - V0)
        )

        denominator = (
            1
            + (
                V0**2
                * np.sin(q * a) ** 2
            )
            / (
                4 * E * (E - V0)
            )
        )

        return 1.0 / denominator

    else:
        # E == V0 limit
        denominator = (
            1
            + TWO_M_OVER_HBAR2
            * V0
            * a**2
            / 4
        )

        return 1.0 / denominator


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================

_NICE = {
    "E": "E",
    "V0": "V₀",
    "a": "a",
    "t": "|t|",
    "T": "T",
    "R": "R",
}


def nice_limit(value):
    return f"{value:.3f}"


def _format_div(value):
    if abs(value) < 1e-12:
        return "0"

    return f"{value:.4g}"


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Experimental Controls")

E = st.sidebar.slider(
    "Particle energy E",
    min_value=float(E_MIN),
    max_value=float(E_MAX),
    value=float(E_DEFAULT),
    step=float(E_STEP),
)

V0 = st.sidebar.slider(
    "Barrier height V₀",
    min_value=float(V0_MIN),
    max_value=float(V0_MAX),
    value=float(V0_DEFAULT),
    step=float(V0_STEP),
)

a = st.sidebar.slider(
    "Barrier width a",
    min_value=float(A_MIN),
    max_value=float(A_MAX),
    value=float(A_DEFAULT),
    step=float(A_STEP),
)


# ============================================================
# CALCULATE COEFFICIENTS
# ============================================================

r, C, D, t, k, kappa = solve_coefficients(
    E,
    V0,
    a,
)

t_abs = abs(t)

T = abs(t) ** 2
R = abs(r) ** 2
RT = R + T

T_analytical = transmission_analytical(
    E,
    V0,
    a,
)


# ============================================================
# SPATIAL GRID
# ============================================================

x = np.linspace(
    X_MIN,
    X_MAX,
    N_POINTS,
)

psi = wavefunction(
    x,
    r,
    C,
    D,
    t,
    k,
    kappa,
    a,
)

psi_real = np.real(psi)
psi_abs = np.abs(psi)


# ============================================================
# MEASUREMENT WINDOW
# ============================================================

x0 = min(
    a + BUFFER,
    X_MAX - 0.5,
)

measurement_mask = x >= x0

xm = x[measurement_mask]

# Use the already calculated t and k.
# This avoids solving the 4x4 system again.
psi_m = t * np.exp(1j * k * xm)

psi_m_real = np.real(psi_m)
psi_m_abs = np.abs(psi_m)


# ============================================================
# MAIN PLOT
# ============================================================

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=False,
    vertical_spacing=0.08,
    row_heights=[0.55, 0.45],
    specs=[
        [{"secondary_y": True}],
        [{"secondary_y": False}],
    ],
)


# ============================================================
# TOP GRAPH
# ============================================================

fig.add_trace(
    go.Scatter(
        x=x,
        y=psi_real,
        mode="lines",
        name="Re(ψ)",
        line=dict(
            width=2,
        ),
    ),
    row=1,
    col=1,
    secondary_y=False,
)


# Transmitted part highlighted in red

transmitted_mask = x >= a

fig.add_trace(
    go.Scatter(
        x=x[transmitted_mask],
        y=psi_real[transmitted_mask],
        mode="lines",
        name="Transmitted Re(ψ)",
        line=dict(
            color=TRANSMITTED_COLOUR,
            width=2,
        ),
    ),
    row=1,
    col=1,
    secondary_y=False,
)


# Absolute wavefunction

fig.add_trace(
    go.Scatter(
        x=x,
        y=psi_abs,
        mode="lines",
        name="|ψ|",
        line=dict(
            dash="dash",
            width=1.3,
        ),
    ),
    row=1,
    col=1,
    secondary_y=False,
)


# Negative |t|

fig.add_hline(
    y=-t_abs,
    line_dash="dot",
    line_width=1.4,
    line_color=TRANSMITTED_COLOUR,
    row=1,
    col=1,
)


# Positive |t|

fig.add_hline(
    y=t_abs,
    line_dash="dot",
    line_width=1.4,
    line_color=TRANSMITTED_COLOUR,
    row=1,
    col=1,
)


# ============================================================
# BARRIER REGION
# ============================================================

fig.add_vrect(
    x0=0,
    x1=a,
    fillcolor=BARRIER_COLOUR,
    line_width=0,
    annotation_text="Barrier",
    annotation_position="top",
    row=1,
    col=1,
)


# ============================================================
# MEASURED REGION
# ============================================================

fig.add_vrect(
    x0=x0,
    x1=X_MAX,
    fillcolor=MEASURED_COLOUR,
    line_width=0,
    annotation_text="Measurement region",
    annotation_position="top left",
    row=1,
    col=1,
)


# ============================================================
# POTENTIAL / ENERGY ON SECONDARY AXIS
# ============================================================

V = np.zeros_like(x)

V[(x >= 0) & (x <= a)] = V0

fig.add_trace(
    go.Scatter(
        x=x,
        y=V,
        mode="lines",
        name="V(x)",
        line=dict(
            width=2,
        ),
    ),
    row=1,
    col=1,
    secondary_y=True,
)


# Energy line

fig.add_hline(
    y=E,
    line_dash="dash",
    line_width=1.5,
    annotation_text="E",
    annotation_position="top right",
    row=1,
    col=1,
    secondary_y=True,
)


# ============================================================
# LOWER MEASUREMENT GRAPH
# ============================================================

fig.add_trace(
    go.Scatter(
        x=xm,
        y=psi_m_real,
        mode="lines",
        name="Measured Re(ψ)",
        line=dict(
            color=TRANSMITTED_COLOUR,
            width=2,
        ),
    ),
    row=2,
    col=1,
)


# Positive |t| line

fig.add_hline(
    y=t_abs,
    line_dash="dot",
    line_width=1.4,
    line_color=TRANSMITTED_COLOUR,
    row=2,
    col=1,
)


# Negative |t| line

fig.add_hline(
    y=-t_abs,
    line_dash="dot",
    line_width=1.4,
    line_color=TRANSMITTED_COLOUR,
    row=2,
    col=1,
)


# ============================================================
# GRAPH PAPER STYLE
# ============================================================

fig.update_xaxes(
    showgrid=True,
    gridcolor=GRID_COLOUR,
    zeroline=True,
    zerolinewidth=1,
    row=1,
    col=1,
)

fig.update_yaxes(
    showgrid=True,
    gridcolor=GRID_COLOUR,
    zeroline=True,
    zerolinewidth=1,
    row=1,
    col=1,
    title_text="Wavefunction",
)

fig.update_yaxes(
    showgrid=True,
    gridcolor=GRID_COLOUR,
    zeroline=True,
    zerolinewidth=1,
    row=1,
    col=1,
    secondary_y=True,
    title_text="Potential / Energy",
)

fig.update_xaxes(
    showgrid=True,
    gridcolor=GRID_COLOUR,
    zeroline=True,
    zerolinewidth=1,
    row=2,
    col=1,
    title_text="Position x",
)

fig.update_yaxes(
    showgrid=True,
    gridcolor=GRID_COLOUR,
    zeroline=True,
    zerolinewidth=1,
    row=2,
    col=1,
    title_text="Re(ψ)",
)


# ============================================================
# FIGURE LAYOUT
# ============================================================

fig.update_layout(
    height=850,
    margin=dict(
        l=60,
        r=60,
        t=50,
        b=60,
    ),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)


# ============================================================
# DISPLAY GRAPH
# ============================================================

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# MEASUREMENT INSTRUCTIONS
# ============================================================

st.subheader("📏 Measurement")

st.info(
    "Measure |t| from the lower graph, then the transmission "
    "coefficient is T = |t|²."
)


# ============================================================
# RESULTS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Barrier height V₀",
        f"{V0:.3f}",
    )

with col2:
    st.metric(
        "Particle energy E",
        f"{E:.3f}",
    )

with col3:
    st.metric(
        "Barrier width a",
        f"{a:.3f}",
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "|t|",
        f"{t_abs:.6f}",
    )

with col2:
    st.metric(
        "T = |t|²",
        f"{T:.6f}",
    )

with col3:
    st.metric(
        "R = |r|²",
        f"{R:.6f}",
    )

with col4:
    st.metric(
        "R + T",
        f"{RT:.6f}",
    )


# ============================================================
# INSTRUCTOR KEY
# ============================================================

if INSTRUCTOR_KEY:

    st.markdown("---")

    st.subheader("🔑 Instructor Key")

    key_col1, key_col2 = st.columns(2)

    with key_col1:

        st.write(
            f"**|t| = {_format_div(t_abs)}**"
        )

        st.write(
            f"**T = {_format_div(T)}**"
        )

        st.write(
            f"**R = {_format_div(R)}**"
        )

        st.write(
            f"**R + T = {_format_div(RT)}**"
        )

    with key_col2:

        st.write(
            f"**Analytical T = "
            f"{_format_div(T_analytical)}**"
        )

        st.write(
            f"**Difference = "
            f"{_format_div(abs(T - T_analytical))}**"
        )


# ============================================================
# THEORY
# ============================================================

with st.expander("📘 Theory"):

    st.markdown("### Region I — Incident and Reflected Waves")

    st.markdown("For \(x < 0\),")

    st.latex(
        r"\psi_I(x) = e^{ikx} + r e^{-ikx}"
    )

    st.markdown("where")

    st.latex(
        r"k = \sqrt{\frac{2mE}{\hbar^2}}"
    )

    st.markdown(
        """
        The first term is the **incident wave**, and the second term is
        the **reflected wave**.
        """
    )

    st.markdown("---")

    st.markdown("### Region II — Inside the Barrier")

    st.markdown("For")

    st.latex(
        r"0 \leq x \leq a"
    )

    st.markdown("the wavefunction is")

    st.latex(
        r"\psi_{II}(x) = C e^{\kappa x} + D e^{-\kappa x}"
    )

    st.markdown("where, for \(E < V_0\),")

    st.latex(
        r"\kappa = \sqrt{\frac{2m(V_0-E)}{\hbar^2}}"
    )

    st.markdown(
        """
        The wavefunction therefore decays exponentially inside the
        **classically forbidden region**.
        """
    )

    st.markdown("---")

    st.markdown("### Region III — Transmitted Wave")

    st.markdown("For \(x > a\),")

    st.latex(
        r"\psi_{III}(x) = t e^{ikx}"
    )

    st.markdown(
        "The magnitude of the transmission amplitude is"
    )

    st.latex(
        r"|t|"
    )

    st.markdown(
        "and the transmission coefficient is"
    )

    st.latex(
        r"T = |t|^2"
    )

    st.markdown("Similarly,")

    st.latex(
        r"R = |r|^2"
    )

    st.markdown(
        "and conservation of probability gives"
    )

    st.latex(
        r"R + T = 1"
    )

    st.markdown(
        "up to numerical precision."
    )

# ============================================================
# PARAMETERS
# ============================================================

with st.expander("⚙️ Simulation Parameters"):

    st.write(
        f"**ℏ:** {HBAR}"
    )

    st.write(
        f"**Mass:** {MASS}"
    )

    st.write(
        f"**2m/ℏ²:** {TWO_M_OVER_HBAR2}"
    )

    st.write(
        f"**Spatial range:** "
        f"{X_MIN} to {X_MAX}"
    )

    st.write(
        f"**Spatial points:** "
        f"{N_POINTS}"
    )

    st.write(
        f"**Measurement buffer:** "
        f"{BUFFER}"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Quantum Tunnelling through a Rectangular Barrier — "
    "Interactive Streamlit Laboratory"
)
