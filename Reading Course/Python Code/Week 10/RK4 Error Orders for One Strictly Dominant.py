import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def equilibrium_metric(p,p_expected): # RMS Error
    metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
    return np.sqrt(metric_squared)

#%% dt or T Comparison

A = np.array([[2,3],
           [4,1]], dtype=float)

B = np.array([[7,5],
           [6,8]], dtype=float)

x1 = .99
x2 = .99

p_actual = np.array([1/2,1/2,1/2,1/2])
T_array = np.logspace(1, 3, 40)
error_values = np.zeros((4,len(T_array))) 

for index in tqdm(range(len(T_array)),bar_format="{bar:30} {n_fmt}/{total_fmt}"):  
    T = T_array[index]

    # Constant dt of .1
    t,solution_matrix = bnn_ode_solver([x1,1-x1],[x2,1-x2],A,B,.1,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    error_values[0,index] = equilibrium_metric(p,p_actual)

    # Constant dt of .01
    t,solution_matrix = bnn_ode_solver([x1,1-x1],[x2,1-x2],A,B,.01,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    error_values[1,index] = equilibrium_metric(p,p_actual)
    
    # Constant dt of .005
    t,solution_matrix = bnn_ode_solver([x1,1-x1],[x2,1-x2],A,B,.005,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    error_values[2,index] = equilibrium_metric(p,p_actual)

    # Constant dt of 3.5
    t,solution_matrix = bnn_ode_solver([x1,1-x1],[x2,1-x2],A,B,3.5,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    error_values[3,index] = equilibrium_metric(p,p_actual)

# Errors
plt.figure(figsize=(10, 6))
plt.plot(np.log10(T_array), np.log10(error_values[0,:]),color="red",label=r"Constant $\Delta t = .1$")
plt.plot(np.log10(T_array), np.log10(error_values[1,:]),color="blue",linestyle='--',label=r"Constant $\Delta t = .01$")
plt.plot(np.log10(T_array), np.log10(error_values[2,:]),color="green",linestyle=':',label=r"Constant $\Delta t = .005$")
plt.plot(np.log10(T_array), np.log10(error_values[3,:]),color="orange",label=r"Constant $\Delta t = 3.5$")
plt.xlabel("$log_{10}(T)$")
plt.ylabel("$log_{10}$(Equilibrium Error)")
plt.title("BNN Error Order")
plt.legend()
plt.grid(True)
plt.show()