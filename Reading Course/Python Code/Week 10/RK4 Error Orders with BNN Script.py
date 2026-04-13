import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

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
  m,n = A.shape
    
  def ode_system(p):
    """
    t - Current Time
    p - Concatenated Probability Array
    """
    p1 = p[:m]
    p2 = p[m:]
        
    # Expected Value Under Current Mixed Strategies
    current_value1 = p1.T @ A @ p2
    current_value2 = p1.T @ B @ p2
        
    # Pure Strategy Payoffs with Opponent's Current Mix
    pure_payoffs1 = A @ p2
    pure_payoffs2 = p1.T @ B
        
    # Phi Function
        # ReLU Activation Function      
    phi_vector1 = np.maximum(pure_payoffs1-current_value1,0)
    phi_vector2 = np.maximum(pure_payoffs2-current_value2,0)
        
    # BNN Dynamics
    dp1_dt = phi_vector1 - p1*phi_vector1.sum()
    dp2_dt = phi_vector2 - p2*phi_vector2.sum()

    return np.concatenate((dp1_dt, dp2_dt))


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
     
     p_vector += step_size*(k1+2*k2+2*k3+k4)/6
     
     p_values[:,index+1] = p_vector
    
  return t_mesh,p_values


def quick_bnn_solver(p1, p2, step_size, T):
    p10 = float(p1[0])
    p20 = float(p2[0])

    N = int(np.ceil(T/step_size)) 
    t_mesh = np.linspace(0,T,N)

    p_values = np.zeros((4, N), dtype=float)
    p_values[:, 0] = [p10,1-p10,p20,1-p20]

    p1 = p10
    p2 = p20

    for index in range(N-1):
        # RK4 for p1 Time Evolution
        k1 = -p1*p1
        p1_1 = p1 + 0.5*step_size*k1
        k2 = -p1_1*p1_1
        p1_2 = p1 + 0.5*step_size*k2
        k3 = -p1_2*p1_2
        p1_3 = p1 + step_size * k3
        k4 = -p1_3*p1_3     
        p1 += h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        
        # RK4 for p1 Time Evolution
        k1 = -p2*p2
        p2_1 = p2 + 0.5*step_size*k1
        k2 = -p2_1*p2_1
        p2_2 = p2 + 0.5*step_size*k2
        k3 = -p2_2*p2_2
        p2_3 = p2 + step_size * k3
        k4 = -p2_3*p2_3
        p2 += h*(k1 + 2*k2 + 2*k3 + k4)/6.0

        p_values[:,index+1] = [p1,1.0-p1,p2,1.0-p2]

    return t_mesh, p_values

def equilibrium_metric(p,p_expected): # RMS Error
    metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
    return np.sqrt(metric_squared)

#%% Runtime Comparison

A = np.array([[1,2],
           [3,4]], dtype=float)

B = np.array([[5,7],
           [6,8]], dtype=float)

x1 = .5
x2 = .5
T = 100
p_actual = np.array([0, 1, 0, 1])

h_values = [1,.1,.01,.001,.0001,.00005]
t_values = np.zeros((2,len(h_values)))
error_values = np.zeros((2,len(h_values)))

for index in range(len(h_values)):
    h = h_values[index]
    
    if h <= 4:
        start = time.time()
        t,solution_matrix = bnn_ode_solver([x1,1-x1],[x2,1-x2],A,B,h,T)
        end = time.time()
        p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
        t_values[0,index] = end-start
        error_values[0,index] = equilibrium_metric(p,p_actual)
    
    start = time.time()
    t,solution_matrix = quick_bnn_solver([x1,1-x1],[x2,1-x2],h,T)
    end = time.time()
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    t_values[1,index] = end-start
    error_values[1,index] = equilibrium_metric(p,p_actual)

# Compare Errors
plt.figure(figsize=(10, 6))
plt.plot(h_values[:4], error_values[0,:4],color="red",label="Normal")
plt.plot(h_values, error_values[1,:],color="blue",label="Quick")
plt.xlabel("Step Size")
plt.xscale("log")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Normal vs Quick Method Errors")
plt.legend()
plt.grid(True)
plt.show()

# Compare Runtimes
plt.figure(figsize=(10, 6))
plt.plot(h_values, t_values[0,:],color="red",label="Normal")
plt.plot(h_values, t_values[1,:],color="blue",label="Quick")
plt.xlabel("Step Size")
plt.ylabel("Runtime")
plt.title("Normal vs Quick Method Runtimes")
plt.legend()
plt.grid(True)
plt.show()

#%% Error Order

h_array = np.logspace(0, -5, 15)

def equilibrium_metric(p,p_expected): # RMS Error
    metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
    return np.sqrt(metric_squared)

metric_array = np.zeros((len(h_array)))
counter = 0
for h in tqdm(h_array):
    t,solution_matrix = quick_bnn_solver([x1,1-x1],[x2,1-x2],h,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    p_actual = np.array([0, 1, 0, 1])
    metric_array[counter] = equilibrium_metric(p,p_actual)
    counter += 1

# Errors
plt.figure(figsize=(10, 6))
plt.plot(np.log10(h_array), np.log10(metric_array),color="red")
plt.xlabel("log10(Step Size)")
plt.ylabel("log10(Error")
plt.title("BNN Error Order")
plt.grid(True)
plt.show()