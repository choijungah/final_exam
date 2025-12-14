import streamlit as st
import matplotlib.pyplot as plt

st.title("Simple Line Plot")

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

fig, ax = plt.subplots()
ax.plot(x, y, marker="o")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("y = x²")
ax.grid(True)

st.pyplot(fig)
