import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 400
L = 1.0
dx = L/N

T = .21
dt = 2e-6
steps = int(T/dt)

viscosity = 1/2
assert(dt<=dx**2/2/viscosity) # Diffusive Condition

x = jnp.linspace(0, L, N, endpoint=False)

# Initial Condition
rho = 1.0 + 0.25*jnp.exp(-200*(x - 0.25)**2) + 0.25*jnp.exp(-200*(x - 0.75)**2)
u = jnp.zeros_like(x)

# Finite Differences [Second Order]
def ddx(f):
    return (jnp.roll(f, -1)-jnp.roll(f, 1))/(2*dx)

def d2dx2(f):
    return (jnp.roll(f,-1)-2*f+jnp.roll(f, 1))/dx**2

gamma = .5
def pressure(rho):
    return rho**gamma

# One Time Step
@jax.jit
def step(rho, u):
    p = pressure(rho)

    # Flow
    u_t = -u*ddx(u) + viscosity*d2dx2(u) - (1/rho)*ddx(p) 
    
    # Density
    rho_t = -ddx(rho*u)

    # Update
    rho_new = rho + dt * rho_t
    u_new = u + dt * u_t
    
    # Minimum Density
    rho_new = jnp.maximum(rho_new, 1e-6)

    return rho_new, u_new

# Run Simulation
plt.figure()

for n in range(steps):
    rho, u = step(rho, u)
    
    c = jnp.sqrt(gamma * rho**(gamma - 1)) 
    max_wave_speed = jnp.max(jnp.abs(u) + c)
    assert(dt <= dx/max_wave_speed)

    if n % 50000 == 0:
        plt.clf()
        plt.plot(np.array(x), np.array(rho), label=r"$\rho$")
        plt.plot(np.array(x), np.array(u), label=r"$v$")
        plt.ylim(-0.5, 1.5)
        plt.title(f"1D compressible Navier-Stokes, t = {n*dt:.4f}")
        plt.xlabel("x")
        plt.legend()
        plt.pause(0.01)

plt.show()