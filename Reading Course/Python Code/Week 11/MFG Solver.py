# %% 1D Mean Field Game Solver with H = 1/2 |grad u|^2

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
Nx = 200
Nt = 5000
L = 1.0
T = 1.0

dx = L / Nx
dt = T / Nt
nu = 0.01

x = np.linspace(0, L, Nx, endpoint=False)

# -----------------------
# Finite differences: periodic boundary
# -----------------------
def ddx(f):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

def d2dx2(f):
    return (np.roll(f, -1) - 2*f + np.roll(f, 1)) / dx**2

# -----------------------
# Hamiltonian and coupling
# -----------------------
def H(p):
    return 0.5 * p**2

def dH_dp(p):
    return p

def W(m):
    return .01*m

def W0(m):
    return .01*m

# -----------------------
# Initial density m0(x)
# -----------------------
m0 = 1.0 + 0.2 * np.exp(-100 * (x - 0.5)**2)

# Normalize density
m0 = m0 / (np.sum(m0) * dx)

# Initial guess for m(t,x)
m = np.tile(m0, (Nt, 1))

# -----------------------
# Fixed point iteration
# -----------------------
n_iter = 30
relax = 0.5

for k in range(n_iter):
    print(f"Iteration {k+1}/{n_iter}")

    # -----------------------
    # Solve HJB backward
    # -----------------------
    u = np.zeros_like(m)

    # Terminal condition
    u[-1, :] = W0(m[-1, :])

    for n in reversed(range(Nt - 1)):
        grad_u = ddx(u[n+1, :])

        u[n, :] = (
            u[n+1, :]
            + dt * (
                nu * d2dx2(u[n+1, :])
                - H(grad_u)
                + W(m[n+1, :])
            )
        )

    # -----------------------
    # Solve Fokker-Planck forward
    # -----------------------
    m_new = np.zeros_like(m)
    m_new[0, :] = m0

    for n in range(Nt - 1):
        grad_u = ddx(u[n, :])
        drift = dH_dp(grad_u)

        flux = drift * m_new[n, :]

        m_new[n+1, :] = (
            m_new[n, :]
            + dt * (
                -ddx(flux)
                + nu * d2dx2(m_new[n, :])
            )
        )

        # Keep density positive and normalized
        m_new[n+1, :] = np.maximum(m_new[n+1, :], 1e-8)
        m_new[n+1, :] = m_new[n+1, :] / (np.sum(m_new[n+1, :]) * dx)

    # Relaxation update
    m = (1 - relax) * m + relax * m_new

# -----------------------
# Plots
# -----------------------
plt.figure()
plt.plot(x, m[0, :], label=r"$m(0,x)$")
plt.plot(x, m[-1, :], label=r"$m(T,x)$")
plt.xlabel("x")
plt.ylabel("density")
plt.title(r"MFG density with $H=\frac{1}{2}|\nabla u|^2$")
plt.legend()
plt.show()

plt.figure()
plt.imshow(m, aspect="auto", origin="lower", extent=[0, L, 0, T])
plt.colorbar(label=r"$m(t,x)$")
plt.xlabel("x")
plt.ylabel("time")
plt.title(r"Density evolution, $H=\frac{1}{2}|\nabla u|^2$")
plt.show()

plt.figure()
plt.imshow(u, aspect="auto", origin="lower", extent=[0, L, 0, T])
plt.colorbar(label=r"$u(t,x)$")
plt.xlabel("x")
plt.ylabel("time")
plt.title(r"Value function $u(t,x)$")
plt.show()