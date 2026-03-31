# %% BNN ODE Solver

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def bnn_ode_solver(p1,p2,A,B,T,N):
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
    value_difference1 = torch.from_numpy(pure_payoffs1-current_value1)
    value_difference2 = torch.from_numpy(pure_payoffs2-current_value2)
        
    phi_tensor1 = torch.relu(value_difference1)
    phi_tensor2 = torch.relu(value_difference2)
        
    phi_vector1 = phi_tensor1.numpy()    
    phi_vector2 = phi_tensor2.numpy()
        
    # BNN Dynamics
    dp1_dt = phi_vector1 - p1*sum(phi_vector1)
    dp2_dt = phi_vector2 - p2*sum(phi_vector2)

    return np.hstack((dp1_dt,dp2_dt))


   # Runge-Kutta 4 Solver
  step_size = T / (N-1)
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
    
# %% Modules to Execute

modules_to_execute = [2.4]
# 0 - Week 7: 2 x 2 Payoff Matrices
# 1 - Mesh Spacing with Week 7 Payoff Matrices
# 2 - Check Run Times vs N for 2 x 2 Matrices from Week 7
# 2.1 - Accruracy of Nash Equilibrium of 2 x 2 system as a function of N
# 2.2 - Accruracy of Nash Equilibrium of Scaled 2 x 2 system as a function of N
# 2.3 - Convergence of Nash Equilibrium of 2 x 2 system as a function of N
# 2.4 - Convergence of Nash Equilibrium of Scaled 2 x 2 system as a function of T
# 3 - 2 x 2 Payoff Matrices with Three Nash Equilibria
# 4 - 3 x 3 Payoff Matrices from Week 7

# %% Week 7: 2 x 2 Payoff Matrices

if 0 in modules_to_execute:
    print("Executing Module 0")
    A = np.array([[7, 2],
               [3, 4]], dtype=float)

    B = np.array([[1, 6],
               [8, 2]], dtype=float)

    initial_p1 = [.5, .5]
    initial_p2 = [.5, .5]

    T = 50
    N = 1000

    t,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
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
    plt.plot(t, x1, label=r"$x^1$")
    plt.plot(t, y1, label=r"$y^1$")
    plt.plot(t, x2, label=r"$x^2$")
    plt.plot(t, y2, label=r"$y^2$")
    plt.xlabel("t")
    plt.ylabel("Probability")
    plt.title("BNN dynamics for the 2x2 game")
    plt.legend()
    plt.grid(True)
    plt.show()

# %% Mesh Spacing with Week 7 Payoff Matrices

if 1 in modules_to_execute:
    print("Executing Module 1")
    A = np.array([[7, 2],
               [3, 4]], dtype=float)

    B = np.array([[1, 6],
               [8, 2]], dtype=float)

    initial_p1 = [.5, .5]
    initial_p2 = [.5, .5]

    T = 50
    N = 51
    
    t,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    x1 = solution_matrix[0,:]
    y1 = solution_matrix[1,:]
    x2 = solution_matrix[2,:]
    y2 = solution_matrix[3,:]

    
    # Plot x1 Trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(t, x1,label=r"$x^1$",color='black')
    plt.scatter(t, x1,color='red')
    plt.xlabel("t")
    plt.ylabel("Probability")
    plt.title("BNN dynamics for the 2x2 game")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot All Trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(t, x1, label=r"$x^1$")
    plt.plot(t, y1, label=r"$y^1$")
    plt.plot(t, x2, label=r"$x^2$")
    plt.plot(t, y2, label=r"$y^2$")
    plt.xlabel("t")
    plt.ylabel("Probability")
    plt.title("BNN dynamics for the 2x2 game")
    plt.legend()
    plt.grid(True)
    plt.show()

# %% Check Run Times vs N for 2 x 2 Matrices from Week 7

import time

if 2 in modules_to_execute:
    print("Executing Module 2")
    
    A = np.array([[7, 2],
               [3, 4]], dtype=float)

    B = np.array([[1, 6],
               [8, 2]], dtype=float)
    
    initial_p1 = [.5, .5]
    initial_p2 = [.5, .5]
    T = 50
    
    N = 2
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")

    N = 20
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")

    N = 200
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")

    N = 2000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    
    N = 20000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("\n")
 
# %% Accruracy of Nash Equilibrium of 2 x 2 system as a function of N

if 2.1 in modules_to_execute:
    print("Executing Module 2.1")
    
    def equilibrium_metric(p,p_expected): # RMS Error
        metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
        return np.sqrt(metric_squared)
    
    A = np.array([[7, 2],
               [3, 4]], dtype=float)

    B = np.array([[1, 6],
               [8, 2]], dtype=float)
    
    initial_p1 = [.5, .5]
    initial_p2 = [.5, .5]
    T = 50
    p_expected = [6/11, 5/11, 1/3, 2/3]
    
    N = 1000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    N = 2000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))

    N = 4000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))

    N = 8000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    N = 16000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    print("\n")

# %% Accruracy of Scaled Nash Equilibrium of 2 x 2 system as a function of N

if 2.2 in modules_to_execute:
    print("Executing Module 2.2")
    
    def equilibrium_metric(p,p_expected): # RMS Error
        metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
        return np.sqrt(metric_squared)
    
    A = np.array([[70, 20],
               [30, 40]], dtype=float)

    B = np.array([[10, 60],
               [80, 20]], dtype=float)
    
    initial_p1 = [.5, .5]
    initial_p2 = [.5, .5]
    T = 50
    p_expected = [6/11, 5/11, 1/3, 2/3]
    
    N = 1000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    N = 2000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))

    N = 4000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))

    N = 8000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    N = 16000
    start = time.time()
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    end = time.time()
    p_final = solution_matrix[:,-1] 
    print("Elapsed time for N = ",N,":", round(end-start,6), "seconds")
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    print("\n")

# %% Convergence of Nash Equilibrium of 2 x 2 system as a function of T

if 2.3 in modules_to_execute:
    print("Executing Module 2.3")
    
    def equilibrium_metric(p,p_expected): # RMS Error
        metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
        return np.sqrt(metric_squared)
    
    A = np.array([[7, 2],
               [3, 4]], dtype=float)

    B = np.array([[1, 6],
               [8, 2]], dtype=float)
    
    initial_p1 = [.5, .5]
    initial_p2 = [.5, .5]
    N = 1000
    p_expected = [6/11, 5/11, 1/3, 2/3]
    
    T = 1
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 5
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))

    T = 50
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))

    T = 200
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 500
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 1000
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
  
    T = 1100
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 1300
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 1500
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 2000
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
# %% Convergence of Nash Equilibrium of Scaled 2 x 2 system as a function of T

if 2.4 in modules_to_execute:
    print("Executing Module 2.4")
    
    def equilibrium_metric(p,p_expected): # RMS Error
        metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
        return np.sqrt(metric_squared)
    
    A = np.array([[7, 2],
               [3, 4]], dtype=float)

    B = np.array([[1, 6],
               [8, 2]], dtype=float)
    
    # Scaling
    A = A
    B = B
    
    initial_p1 = [.5, .5]
    initial_p2 = [.5, .5]
    N = 1000
    p_expected = [6/11, 5/11, 1/3, 2/3]
    
    T = 1
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 5
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))

    T = 50
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))

    T = 200
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 500
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 1000
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
  
    T = 1100
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 1300
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 1500
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    T = 2000
    _,solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    p_final = solution_matrix[:,-1] 
    print("T = ",T)
    print("RMS Equilibrium Error: ",equilibrium_metric(p_final,p_expected))
    
    # %% 2 x 2 Payoff Matrices with Three Nash Equilibria

if 3 in modules_to_execute:
    print("Executing Module 3")
    A = np.array([[1, 0],
                   [0, 2]], dtype=float)
    
    B = np.array([[2, 0],
                   [0, 1]], dtype=float)
    
    T = 50
    N = 1000
    
    initial_p1 = [.25, .75]
    initial_p2 = [.25, .75]
    
    solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    t = solution_matrix[0,:]
    x1 = solution_matrix[1,:]
    y1 = solution_matrix[2,:]
    x2 = solution_matrix[3,:]
    y2 = solution_matrix[4,:]
    
    # Print final state
    print("Final approximate strategy profile:")
    print(f"Player 1: x1 = {x1[-1]:.6f}, y1 = {y1[-1]:.6f}")
    print(f"Player 2: x2 = {x2[-1]:.6f}, y2 = {y2[-1]:.6f}")
    
    # Plot trajectories
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, x1, label=r"$x^1$")
    plt.plot(t, y1, label=r"$y^1$")
    plt.plot(t, x2, label=r"$x^2$")
    plt.plot(t, y2, label=r"$y^2$")
    plt.xlabel("t")
    plt.ylabel("Probability")
    plt.title("BNN dynamics for the 2x2 game")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    initial_p1 = [.5, .5]
    initial_p2 = [.5, .5]
    
    solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    t = solution_matrix[0,:]
    x1 = solution_matrix[1,:]
    y1 = solution_matrix[2,:]
    x2 = solution_matrix[3,:]
    y2 = solution_matrix[4,:]
    
    # Print final state
    print("Final approximate strategy profile:")
    print(f"Player 1: x1 = {x1[-1]:.6f}, y1 = {y1[-1]:.6f}")
    print(f"Player 2: x2 = {x2[-1]:.6f}, y2 = {y2[-1]:.6f}")
    
    # Plot trajectories
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, x1, label=r"$x^1$")
    plt.plot(t, y1, label=r"$y^1$")
    plt.plot(t, x2, label=r"$x^2$")
    plt.plot(t, y2, label=r"$y^2$")
    plt.xlabel("t")
    plt.ylabel("Probability")
    plt.title("BNN dynamics for the 2x2 game")
    plt.legend()
    plt.grid(True)
    plt.show()

# %% 3 x 3 Payoff Matrices from Week 7

if 4 in modules_to_execute:
    print("Executing Module 4")
    A = np.array([
        [4, 1, -3],
        [2, 4, -3],
        [1, 2,  4]
    ], dtype=float)
    
    B = np.array([
        [ 4, 1, 3],
        [ 2, 3, 1],
        [-3, 4, 2]
    ], dtype=float)
    
    T = 50
    N = 1000
    
    initial_p1 = [.3,.3,.4]
    initial_p2 = [.3,.3,.4]
    
    solution_matrix = bnn_ode_solver(initial_p1,initial_p2,A,B,T,N)
    t = solution_matrix[0,:]
    x1 = solution_matrix[1,:]
    y1 = solution_matrix[2,:]
    z1 = solution_matrix[3,:]
    x2 = solution_matrix[4,:]
    y2 = solution_matrix[5,:]
    z2 = solution_matrix[6,:]
    
    # Print final state
    print("Final approximate strategy profile:")
    print(f"Player 1: x1 = {x1[-1]:.6f}, y1 = {y1[-1]:.6f}, z1 = {z1[-1]:.6f}")
    print(f"Player 2: x2 = {x2[-1]:.6f}, y2 = {y2[-1]:.6f}, z2 = {z2[-1]:.6f}")
    
    # Plot trajectories
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, x1, label=r"$x^1$")
    plt.plot(t, y1, label=r"$y^1$")
    plt.plot(t, z1, label=r"$z^1$")
    plt.plot(t, x2, label=r"$x^2$")
    plt.plot(t, y2, label=r"$y^2$")
    plt.plot(t, z2, label=r"$2^1$")
    plt.xlabel("t")
    plt.ylabel("Probability")
    plt.title("BNN dynamics for the 3x3 game")
    plt.legend()
    plt.grid(True)
    plt.show()