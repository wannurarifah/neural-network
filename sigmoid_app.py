
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Sigmoid Activation Function",
    page_icon="📈",
    layout="wide"
)

st.title("Sigmoid Activation Function")
st.write("Interactive visualisation of the Sigmoid activation function.")

st.markdown("---")

# -------------------------------------------------
# Introduction
# -------------------------------------------------
st.subheader("1. Introduction")

st.write("""
Sigmoid is an activation function that converts input values
into outputs between 0 and 1. It produces an S-shaped curve
and is commonly used in neural networks.
""")

st.latex(r"f(x) = \frac{1}{1+e^{-x}}")

# -------------------------------------------------
# Interactive Input
# -------------------------------------------------
st.subheader("2. Interactive Sigmoid Calculator")

x_input = st.slider(
    "Select an input value (x)",
    min_value=-10.0,
    max_value=10.0,
    value=2.0,
    step=0.5
)

# Calculate Sigmoid output
sigmoid_output = 1 / (1 + np.exp(-x_input))

col1, col2 = st.columns(2)

with col1:
    st.metric("Input Value (x)", f"{x_input:.2f}")

with col2:
    st.metric("Sigmoid Output f(x)", f"{sigmoid_output:.4f}")

# Explain the result
if x_input < 0:
    st.info("The input is negative. Therefore, the output is closer to 0.")
elif x_input == 0:
    st.info("The input is zero. Therefore, the output is exactly 0.5.")
else:
    st.success("The input is positive. Therefore, the output is closer to 1.")

# -------------------------------------------------
# Sigmoid Graph
# -------------------------------------------------
st.subheader("3. Sigmoid Activation Function Graph")

# Generate x values
x = np.linspace(-10, 10, 400)

# Calculate Sigmoid values
y = 1 / (1 + np.exp(-x))

# Create graph
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, y, label="Sigmoid: f(x) = 1/(1+e^(-x))",
        linewidth=2)

# Show selected input point
ax.scatter(
    x_input,
    sigmoid_output,
    color="red",
    s=80,
    label=f"Selected Point ({x_input:.1f}, {sigmoid_output:.4f})"
)

ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(1, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("Sigmoid Activation Function")
ax.set_ylim(-0.1, 1.1)
ax.grid(True)
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Interpretation
# -------------------------------------------------
st.subheader("4. Graph Interpretation")

st.write("""
- When x is negative, the output approaches 0.
- When x is positive, the output approaches 1.
- When x = 0, the output is exactly 0.5.
- The graph has an S-shaped curve.
- The output values are always between 0 and 1.
""")

st.markdown("---")
st.caption("Chapter 4: Neural Networks | Sigmoid Activation Function")