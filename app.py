import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ==========================
# PAGE SETTINGS
# ==========================

st.set_page_config(
    page_title="Quantum Entanglement Explorer",
    layout="wide"
)

# ==========================
# LOAD MODEL & DATA
# ==========================

model = joblib.load("entanglement_model.pkl")

df = pd.read_excel("comparison.xlsx")

pivot = df.pivot(
    index="Bond Distance",
    columns="Bond Angle",
    values="Entanglement"
)

emin = df["Entanglement"].min()
emax = df["Entanglement"].max()

# ==========================
# TITLE
# ==========================

st.title("🔬 Quantum Entanglement Explorer")

st.markdown(
"""
Machine Learning Assisted Prediction of Molecular Entanglement in Water Molecules
"""
)

# ==========================
# SIDEBAR CONTROLS
# ==========================

st.sidebar.header("Geometry Controls")

angle = st.sidebar.slider(
    "Bond Angle (°)",
    float(df["Bond Angle"].min()),
    float(df["Bond Angle"].max()),
    104.5
)

distance = st.sidebar.slider(
    "Bond Distance (Å)",
    float(df["Bond Distance"].min()),
    float(df["Bond Distance"].max()),
    0.96
)

# ==========================
# PREDICTION
# ==========================

sample = pd.DataFrame({
    "Bond Angle": [angle],
    "Bond Distance": [distance]
})

ent = model.predict(sample)[0]

st.metric(
    "Predicted Entanglement ξ",
    f"{ent:.6f}"
)

# ==========================
# LAYOUT
# ==========================

col1, col2 = st.columns(2)

# ====================================
# HEATMAP
# ====================================

with col1:

    fig1, ax1 = plt.subplots(figsize=(6,5))

    extent = [
        pivot.columns.min(),
        pivot.columns.max(),
        pivot.index.min(),
        pivot.index.max()
    ]

    im = ax1.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="plasma"
    )

    ax1.scatter(
        angle,
        distance,
        s=350,
        marker="*",
        color="white"
    )

    ax1.scatter(
        104.5,
        0.96,
        s=120,
        color="cyan"
    )

    ax1.set_title("Entanglement Landscape")
    ax1.set_xlabel("Bond Angle (°)")
    ax1.set_ylabel("Bond Distance (Å)")

    fig1.colorbar(
        im,
        ax=ax1,
        label="Entanglement ξ"
    )

    st.pyplot(fig1)

# ====================================
# MOLECULE
# ====================================

with col2:

    theta = np.radians(angle)

    O = np.array([0, 0])

    H1 = np.array([distance, 0])

    H2 = np.array([
        distance*np.cos(theta),
        distance*np.sin(theta)
    ])

    fig2, ax2 = plt.subplots(figsize=(6,5))

    ax2.plot(
        [O[0], H1[0]],
        [O[1], H1[1]],
        linewidth=5
    )

    ax2.plot(
        [O[0], H2[0]],
        [O[1], H2[1]],
        linewidth=5
    )

    ax2.scatter(
        O[0],
        O[1],
        s=2500
    )

    ax2.scatter(
        H1[0],
        H1[1],
        s=1200
    )

    ax2.scatter(
        H2[0],
        H2[1],
        s=1200
    )

    ax2.text(O[0], O[1], "O")
    ax2.text(H1[0], H1[1], "H")
    ax2.text(H2[0], H2[1], "H")

    ax2.set_aspect("equal")

    ax2.set_title(
        f"Water Geometry\nξ = {ent:.6f}"
    )

    st.pyplot(fig2)

# ==========================
# MODEL INFO
# ==========================

st.markdown("---")

st.subheader("Model Information")

st.write("Random Forest Regression")

st.write("R² Score: 0.9221")

st.write("Training Data: 450 molecular geometries")
