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

# Picard Parameters
picard_steps = 20
tol = 1e-6
learning_rate = 0.5

# Terminal Density Condition
rho_T = 1.0 + 0.25*jnp.exp(-200*(x - 0.25)**2) + 0.25*jnp.exp(-200*(x - 0.75)**2)
u = jnp.zeros_like(x)

# Initial Flow Condition
def h(rho):
    return rho

# Initial Guess for Density
rho_path = jnp.tile(rho_T[None, :],(steps+1,1))

# Finite Differences [Second Order]
def ddx(f):
    return (jnp.roll(f, -1)-jnp.roll(f, 1))/(2*dx)

def d2dx2(f):
    return (jnp.roll(f,-1)-2*f+jnp.roll(f, 1))/dx**2

gamma = .5
def pressure(rho):
    return rho**gamma

# Forward Solve for Flow
@jax.jit
def step_v_forward(rho_frozen, v):
    p = pressure(rho_frozen)

    # Flow
    v_t = -v*ddx(v) + viscosity*d2dx2(v) - ddx(p) 
    
    v_new = v + dt * v_t
    return v_new

def solve_v_forward(rho_path):
    v0 = ddx(h(rho_path[0]))
    v = v0
    v_list = [v]
    
    for n in range(steps):
        v = step_v_forward(rho_path[n],v)
        v_list.append(v)

    return jnp.stack(v_list)

# Backward Solve for Density
@jax.jit
def step_rho_backward(rho, v_frozen):
    rho_t = viscosity*d2dx2(rho) + ddx(rho*v_frozen)

    rho_old = rho-dt*rho_t
    rho_old = jnp.maximum(rho_old, 1e-6)

    return rho_old

def solve_rho_backward(v_path):
    rho = rho_T
    rho_reverse = [rho]

    for n in range(steps, 0, -1):
        rho = step_rho_backward(rho, v_path[n])
        rho_reverse.append(rho)

    return jnp.stack(rho_reverse[::-1])

# Picard Iteration
for k in range(picard_steps):
    old_rho_path = rho_path

    # Step 1: Solve for v^(k+1) 
    v_path = solve_v_forward(rho_path)

    # Step 2: Solve rho^(k+1) backward using v^(k+1)
    new_rho_path = solve_rho_backward(v_path)

    # Learning Rate helps convergence
    rho_path = learning_rate*new_rho_path + (1-learning_rate)*old_rho_path

    error = jnp.max(jnp.abs(rho_path - old_rho_path))

    print(f"Picard iteration {k+1}: error = {float(error):.3e}")

    if error < tol:
        print("Converged.")
        break

# Plot Solution
plt.figure()

plt.figure()
plt.plot(x, rho_path[0, :], label=r"$m(0,x)$")
plt.plot(x, rho_path[-1, :], label=r"$m(T,x)$")
plt.xlabel("x")
plt.ylabel("density")
plt.title(r"MFG density with $H=\frac{1}{2}|\nabla u|^2$")
plt.legend()
plt.show()

plt.figure()
plt.imshow(rho_path, aspect="auto", origin="lower", extent=[0, L, 0, T])
plt.colorbar(label=r"$m(t,x)$")
plt.xlabel("x")
plt.ylabel("time")
plt.title(r"Density evolution, $H=\frac{1}{2}|\nabla u|^2$")
plt.show()

plt.figure()
plt.imshow(v_path, aspect="auto", origin="lower", extent=[0, L, 0, T])
plt.colorbar(label=r"$v(t,x)$")
plt.xlabel("x")
plt.ylabel("time")
plt.title(r"Value function $u(t,x)$")
plt.show()