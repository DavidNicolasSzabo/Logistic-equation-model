
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Parameters for logistic equation dx/dt = a*x - b*x^2
a = 0.01
b = 0.0001
K = a / b   # Carrying capacity = 100

# ---- SYMBOLIC SOLUTION ----
t = sp.symbols('t', real=True)
x0, x1 = sp.symbols('x0 x1', positive=True)
r = a

# General logistic solution with x(0) = x0
x_t = (K * x0 * sp.exp(r * t)) / (K + x0 * (sp.exp(r * t) - 1))
print("Symbolic solution:")
sp.pprint(x_t, use_unicode=True)

# Time to grow from x0 to x1
eq = sp.Eq(x_t, x1)
t_expr = sp.solve(eq, t)[0]
t_expr = sp.simplify(t_expr)
print("\nTime to grow from x0 to x1:")
sp.pprint(t_expr, use_unicode=True)

# Example numeric evaluation
x0_val, x1_val = 10, 80
t_val = float(t_expr.subs({x0: x0_val, x1: x1_val}))
print(f"\nExample: time to grow from {x0_val} to {x1_val} ≈ {t_val:.2f} time units")

# ---- SLOPE FIELD + SOLUTION CURVES ----
t_vals = np.linspace(0, 600, 400)
x_grid = np.linspace(0, 160, 400)
T, X = np.meshgrid(np.linspace(0, 600, 25), np.linspace(0, 160, 25))
DX = a * X - b * X**2

# Normalize arrows for slope field
U = 1 / np.sqrt(1 + DX**2)
V = DX / np.sqrt(1 + DX**2)

plt.figure(figsize=(10,6))
plt.title("Slope Field and Solutions: a= 0.01 and b=0.0001")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.xlim(0, 600)
plt.ylim(0, 160)

# Plot slope field
plt.quiver(T, X, U, V, angles='xy', scale_units='xy', scale=20, width=0.003)

# Plot several solution curves
initial_conditions = [1, 5, 10, 20, 50, 120, 140]
for x0_i in initial_conditions:
    x_t_num = (K * x0_i * np.exp(a * t_vals)) / (K + x0_i * (np.exp(a * t_vals) - 1))
    plt.plot(t_vals, x_t_num, label=f"x₀={x0_i}")

plt.axhline(K, color='black', linestyle='--', linewidth=1, label="K=100")
plt.legend()
plt.tight_layout()
plt.show()
