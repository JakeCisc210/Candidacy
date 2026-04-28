# %% JAX 1D viscous compressible Navier-Stokes

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
N = 400
L = 1.0
dx = L / N

T = 0.15
dt = 2e-5
steps = int(T / dt)

gamma = 1.4
K = 1.0
mu = 0.002

x = jnp.linspace(0, L, N, endpoint=False)

# Initial condition
rho = 1.0 + 0.25 * jnp.exp(-200 * (x - 0.5)**2)
u = jnp.zeros_like(x)

# -----------------------
# Finite differences
# -----------------------
def ddx(f):
    return (jnp.roll(f, -1) - jnp.roll(f, 1)) / (2 * dx)

def d2dx2(f):
    return (jnp.roll(f, -1) - 2*f + jnp.roll(f, 1)) / dx**2

def pressure(rho):
    return K * rho**gamma

# -----------------------
# One time step
# -----------------------
@jax.jit
def step(rho, u):
    p = pressure(rho)

    # rho_t + (rho u)_x = 0
    rho_t = -ddx(rho * u)

    # u_t + u u_x + (1/rho) p_x = mu u_xx
    u_t = -u * ddx(u) - (1 / rho) * ddx(p) + mu * d2dx2(u)

    rho_new = rho + dt * rho_t
    u_new = u + dt * u_t

    rho_new = jnp.maximum(rho_new, 1e-6)

    return rho_new, u_new

# -----------------------
# Run simulation
# -----------------------
plt.figure()

for n in range(steps):
    rho, u = step(rho, u)

    if n % 500 == 0:
        plt.clf()
        plt.plot(np.array(x), np.array(rho), label=r"$\rho$")
        plt.plot(np.array(x), np.array(u), label=r"$u$")
        plt.ylim(-0.5, 1.5)
        plt.title(f"1D compressible Navier-Stokes, t = {n*dt:.4f}")
        plt.xlabel("x")
        plt.legend()
        plt.pause(0.01)

plt.show()