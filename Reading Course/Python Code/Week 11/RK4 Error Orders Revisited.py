import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

def ode_system(t,p,A,B):
  """
  t - Current Time
  p - Concatenated Probability Array
  """
  p1 = p[range(2)]
  p2 = p[range(2,4)]
      
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
      
  phi_vector1 = np.maximum(value_difference1,0)*np.maximum(value_difference1,0)
  phi_vector2 = np.maximum(value_difference2,0)*np.maximum(value_difference2,0)
      
  # BNN Dynamics
  dp1_dt = phi_vector1 - p1*sum(phi_vector1)
  dp2_dt = phi_vector2 - p2*sum(phi_vector2)

  return np.hstack((dp1_dt,dp2_dt))

#%% New ODE Solver System

def bnn_ode_solver(p1,p2,A,B,h,T):
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
    
  def ode_system_2(p):
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
        
    phi_vector1 = np.maximum(value_difference1,0)*np.maximum(value_difference1,0)
    phi_vector2 = np.maximum(value_difference2,0)*np.maximum(value_difference2,0)
        
    # BNN Dynamics
    dp1_dt = phi_vector1 - p1*sum(phi_vector1)
    dp2_dt = phi_vector2 - p2*sum(phi_vector2)

    return np.hstack((dp1_dt,dp2_dt))


   # Runge-Kutta 4 Solver
  N = int(np.ceil(T/h))
  t_mesh = np.linspace(0,T,N+1)
  step_size = T/N
  
  p_values = np.zeros((m+n,N+1))
  p_values[:,0] = np.hstack((p1,p2))
    
  for index in range(N):
     p_vector = p_values[:,index]
        
     k1 = ode_system_2(p_vector)
     k2 = ode_system_2(p_vector+step_size*k1/2)
     k3 = ode_system_2(p_vector+step_size*k2/2)
     k4 = ode_system_2(p_vector+step_size*k3)
     
     p_values[:,index+1] = p_vector + step_size*(k1+2*k2+2*k3+k4)/6
    
  return t_mesh,p_values,step_size

def equilibrium_metric(p,p_expected): # RMS Error
    metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
    return np.sqrt(metric_squared)

#%% Error Order

A = np.array([[2,3],
           [4,1]], dtype=float)

B = np.array([[7,5],
           [6,8]], dtype=float)

p1 = [1/3,2/3]
p2 = [1/3,2/3]
T = 10

h_array = np.logspace(-1, -3, 100)

sol = solve_ivp(
    ode_system, 
    (0,T), 
    np.hstack((p1,p2)), 
    args=(A, B), 
    method='RK45', 
    t_eval=[T],
    rtol=1e-12, 
    atol=1e-14
)

# Exact solutions
x1 = sol.y[0]
y1 = sol.y[1]
x2 = sol.y[2]
y2 = sol.y[3]
p_actual = np.array([x1[-1], y1[-1], x2[-1], y2[-1]])

metric_array = np.zeros((len(h_array)))
dt_array = np.zeros((len(h_array)))

counter = 0
for h in tqdm(h_array,bar_format="{bar:30} {n_fmt}/{total_fmt}"): 
    t,solution_matrix,h_real = bnn_ode_solver(p1,p2,A,B,h,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])    
    metric_array[counter] = equilibrium_metric(p,p_actual)
    dt_array[counter] = h_real
    counter += 1

# Errors
plt.figure(figsize=(10, 6))
plt.plot(np.log10(dt_array), np.log10(metric_array),color="red",label="Error")
plt.plot(np.log10(dt_array), -3+4*np.log10(dt_array),label='O($h^{-4}$)', color='black',linestyle='--',)
plt.legend()
plt.xlabel("log10(Step Size)")
plt.ylabel("log10(Error)")
plt.title("BNN Error Order")
plt.grid(True)
plt.show()

# Probability Values
fig, ax = plt.subplots()
ax.imshow(np.zeros((2,2)),extent=[0,2,2,0],alpha=0)

ax.set_xticks(np.arange(0, 3, 1))
ax.set_yticks(np.arange(0, 3, 1))
ax.grid(color="black", linewidth=1)

ax.set_xticks(np.arange(2)+0.5, minor=True)
ax.set_yticks(np.arange(2) + 0.5, minor=True)
ax.set_xticklabels([f"Strategy {strat}" for strat in {"A","B"}], minor=True,color="blue")
ax.set_yticklabels([f"Strategy {i+1}" for i in range(A.shape[0])], minor=True,color="red")
ax.tick_params(axis='x', which='minor',labeltop=True, labelbottom=False) 

ax.tick_params(which="major", labelbottom=False, labelleft=False)
ax.tick_params(which="minor", length=0)
plt.scatter(2*solution_matrix[2,:],2*solution_matrix[0,:],color="black")