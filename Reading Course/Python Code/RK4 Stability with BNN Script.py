import numpy as np
import matplotlib.pyplot as plt

# Stability function for classical RK4
def R(z):
    return 1 + z + z**2/2 + z**3/6 + z**4/24

# Grid in the complex plane
x = np.linspace(-4, 2, 800)
y = np.linspace(-4, 4, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Absolute stability condition: |R(z)| <= 1
stab = np.abs(R(Z))

# Plot boundary and stable region
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, stab, levels=[0, 1], alpha=0.8)
plt.contour(X, Y, stab, levels=[1], linewidths=2)

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel(r'Re$(z)$')
plt.ylabel(r'Im$(z)$')
plt.title('Region of Absolute Stability for Classical RK4')
plt.gca().set_aspect('equal')
plt.show()


#%% New ODE Solver System

def bnn_ode_solver(p1,p2,A,B,step_size,T):
  """
    Initial Probability Arrays:
        p1 - Initial Player 1 Probability Array
        p2 - Initial Player 2 Probability Array
    
    Payoff Matrices:
        A - Player 1 Payoff Matrix
        B - Player 2 Payoff Matrix
        
    T - Time Period
    N - Number of Time Steps
    """
  m = np.size(A,0)
  n = np.size(A,1) 
    
  def ode_system(p):
    """
    t - Current Time
    p - Concatenated Probability Array
    """
    p1 = p[range(m)]
    p2 = p[range(m,n+m)]
        
    # Expected Value Under Current Mixed Strategies
    current_value1 = p1.T @ A @ p2
    current_value2 = p1.T @ B @ p2
        
    # Pure Strategy Payoffs with Opponent's Current Mix
    pure_payoffs1 = A @ p2
    pure_payoffs2 = p1.T @ B
        
    # Phi Function
        # ReLU Activation Function
    value_difference1 = (pure_payoffs1-current_value1)
    value_difference2 = (pure_payoffs2-current_value2)
        
    phi_vector1 = np.maximum(value_difference1,0)
    phi_vector2 = np.maximum(value_difference2,0)
        
    # BNN Dynamics
    dp1_dt = phi_vector1 - p1*sum(phi_vector1)
    dp2_dt = phi_vector2 - p2*sum(phi_vector2)

    return np.hstack((dp1_dt,dp2_dt))


   # Runge-Kutta 4 Solver
  N = int(np.ceil(T/step_size))
  t_mesh = np.linspace(0,T,N)
  p_values = np.zeros((m+n,N))
  p_values[:,0] = np.hstack((p1,p2))
    
  for index in range(N-1):
     p_vector = p_values[:,index]
        
     k1 = ode_system(p_vector)
     k2 = ode_system(p_vector+step_size*k1/2)
     k3 = ode_system(p_vector+step_size*k2/2)
     k4 = ode_system(p_vector+step_size*k3)
     
     # Stay Within Probabilitic Bounds
     new_value = p_vector + step_size*(k1+2*k2+2*k3+k4)/6
     new_vector = np.maximum(new_value,0)
     new_vector = np.minimum(new_value,1)
     
     p_values[:,index+1] = new_vector
    
  return t_mesh,p_values

#%% Simple BNNs

A = np.array([[6,-2],
           [-4,5]], dtype=float)

B = -A

x1 = 1
x2 = 1
h = .01
T = 100

t,solution_matrix = bnn_ode_solver([x1,1-x1],[x2,1-x2],A,B,h,T)
x1 = solution_matrix[0,:]
y1 = solution_matrix[1,:]
x2 = solution_matrix[2,:]
y2 = solution_matrix[3,:]

# Print final state
print("Final approximate strategy profile:")
print(f"Player 1: x1 = {x1[-1]:.6f}, y1 = {y1[-1]:.6f}")
print(f"Player 2: x2 = {x2[-1]:.6f}, y2 = {y2[-1]:.6f}")

# Plot trajectories
plt.figure(figsize=(10, 6))
plt.plot(t, x1, label=r"$x^1$", color='red')
plt.plot(t, y1, label=r"$y^1$", color='blue')
plt.plot(t, x2, label=r"$x^2$",linestyle='--', color='yellow')
plt.plot(t, y2, label=r"$y^2$",linestyle='--', color='cyan')
plt.xlabel("t")
plt.ylabel("Probability")
plt.title("BNN dynamics for the 2x2 game")
plt.legend()
plt.grid(True)
plt.show()

#%% Error as a function of h

A = np.array([[6,-2],
           [-4,5]], dtype=float)

B = -A

x1 = .99
x2 = .99
T = 100
h_array = np.linspace(.001, .01, 20)

def equilibrium_metric(p,p_expected): # RMS Error
    metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
    return np.sqrt(metric_squared)

metric_array = np.zeros((len(h_array)))
counter = 0
for h in h_array:
    t,solution_matrix = bnn_ode_solver([x1,1-x1],[x2,1-x2],A,B,h,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    p_actual = np.array([9/17, 8/17, 7/17, 10/17])
    metric_array[counter] = equilibrium_metric(p,p_actual)
    counter += 1

# Plot trajectories
plt.figure(figsize=(10, 6))
plt.plot(h_array, metric_array, label="Error",color="black")
plt.plot(np.array([.696,.696]),np.array([0,.1]), label="h=.696",linestyle='--', color='green')
plt.xlabel("Step Size")
plt.ylabel("Equilibrium Error")
plt.title("BNN Accuracy as Function of Step Size")
plt.legend()
plt.grid(True)
plt.show()