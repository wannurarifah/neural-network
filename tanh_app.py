
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Tanh Activation Function",
    page_icon="📈",
    layout="wide"
)

st.title("Tanh Activation Function")
st.write("Interactive visualisation of the Hyperbolic Tangent (Tanh).")

st.markdown("---")

# -------------------------------------------------
# Introduction
# -------------------------------------------------
st.subheader("1. Introduction")

st.write("""
Tanh (Hyperbolic Tangent) is an activation function
that converts input values into outputs between -1 and 1.
It produces an S-shaped curve and is centred around zero.
""")

st.latex(r"f(x) = \tanh(x)")

# -------------------------------------------------
# Interactive Input
# -------------------------------------------------
st.subheader("2. Interactive Tanh Calculator")

x_input = st.slider(
    "Select an input value (x)",
    min_value=-10.0,
    max_value=10.0,
    value=2.0,
    step=0.5
)

# Calculate Tanh output
tanh_output = np.tanh(x_input)

col1, col2 = st.columns(2)

with col1:
    st.metric("Input Value (x)", f"{x_input:.2f}")

with col2:
    st.metric("Tanh Output f(x)", f"{tanh_output:.4f}")

# Explain the result
if x_input < 0:
    st.info("The input is negative. Therefore, the Tanh output is negative.")
elif x_input == 0:
    st.info("The input is zero. Therefore, the Tanh output is exactly 0.")
else:
    st.success("The input is positive. Therefore, the Tanh output is positive.")

# -------------------------------------------------
# Tanh Graph
# -------------------------------------------------
st.subheader("3. Tanh Activation Function Graph")

# Generate x values
x = np.linspace(-10, 10, 400)

# Calculate Tanh values
y = np.tanh(x)

# Create graph
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, y, label="Tanh: f(x) = tanh(x)", linewidth=2)

# Show selected input point
ax.scatter(
    x_input,
    tanh_output,
    color="red",
    s=80,
    label=f"Selected Point ({x_input:.1f}, {tanh_output:.4f})"
)

ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(1, color="black", linewidth=0.8)
ax.axhline(-1, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("Hyperbolic Tangent (Tanh) Activation Function")
ax.set_ylim(-1.1, 1.1)
ax.grid(True)
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Interpretation
# -------------------------------------------------
st.subheader("4. Graph Interpretation")

st.write("""
- When x is negative, the output approaches -1.
- When x is positive, the output approaches 1.
- When x = 0, the output is exactly 0.
- The graph produces an S-shaped curve.
- The output values are always between -1 and 1.
""")

st.markdown("---")
st.caption("Chapter 4: Neural Networks | Tanh Activation Function")