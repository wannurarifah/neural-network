
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="ReLU Activation Function",
    page_icon="📈",
    layout="wide"
)

st.title("ReLU Activation Function")
st.write("Interactive visualisation of the Rectified Linear Unit (ReLU).")

st.markdown("---")

# -------------------------------------------------
# Introduction
# -------------------------------------------------
st.subheader("1. Introduction")

st.write("""
ReLU (Rectified Linear Unit) is an activation function
that converts negative input values to zero while
allowing positive input values to remain unchanged.
""")

st.latex(r"f(x) = \max(0,x)")

# -------------------------------------------------
# Interactive Input
# -------------------------------------------------
st.subheader("2. Interactive ReLU Calculator")

x_input = st.slider(
    "Select an input value (x)",
    min_value=-10.0,
    max_value=10.0,
    value=2.0,
    step=0.5
)

# Calculate ReLU output
relu_output = max(0, x_input)

col1, col2 = st.columns(2)

with col1:
    st.metric("Input Value (x)", f"{x_input:.2f}")

with col2:
    st.metric("ReLU Output f(x)", f"{relu_output:.2f}")

# Explain the result
if x_input < 0:
    st.info("The input is negative. Therefore, ReLU converts it to 0.")
elif x_input == 0:
    st.info("The input is zero. Therefore, the output is also 0.")
else:
    st.success("The input is positive. Therefore, ReLU keeps the original value.")

# -------------------------------------------------
# ReLU Graph
# -------------------------------------------------
st.subheader("3. ReLU Activation Function Graph")

# Generate x values
x = np.linspace(-10, 10, 400)

# Calculate ReLU values
y = np.maximum(0, x)

# Create graph
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, y, label="ReLU: f(x) = max(0,x)", linewidth=2)

# Show selected input point
ax.scatter(x_input, relu_output, color="red", s=80,
           label=f"Selected Point ({x_input}, {relu_output})")

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("Rectified Linear Unit (ReLU)")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Interpretation
# -------------------------------------------------
st.subheader("4. Graph Interpretation")

st.write("""
- When x is negative, the output remains at zero.
- When x is positive, the output increases linearly with x.
- The graph changes direction at x = 0.
- ReLU produces non-negative output values.
""")

st.markdown("---")
st.caption("Chapter 4: Neural Networks | ReLU Activation Function")