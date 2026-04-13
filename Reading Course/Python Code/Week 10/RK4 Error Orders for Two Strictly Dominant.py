import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% New ODE Solver System

def quick_bnn_solver(p1, p2, step_size, T):
    p10 = float(p1[0])
    p20 = float(p2[0])

    N = int(np.ceil(T/step_size))+1
    t_mesh = np.arange(N)*step_size

    p_values = np.zeros((4, N), dtype=float)
    p_values[:, 0] = [p10,1-p10,p20,1-p20]

    p1 = p10
    p2 = p20

    for index in range(N-1):
        # RK4 for p1 Time Evolution
        k1 = -2*p1*p1
        p1_1 = p1 + 0.5*step_size*k1
        k2 = -2*p1_1*p1_1
        p1_2 = p1 + 0.5*step_size*k2
        k3 = -2*p1_2*p1_2
        p1_3 = p1 + step_size * k3
        k4 = -2*p1_3*p1_3     
        p1 += step_size*(k1 + 2*k2 + 2*k3 + k4)/6.0
        
        # RK4 for p1 Time Evolution
        k1 = -2*p2*p2
        p2_1 = p2 + 0.5*step_size*k1
        k2 = -2*p2_1*p2_1
        p2_2 = p2 + 0.5*step_size*k2
        k3 = -2*p2_2*p2_2
        p2_3 = p2 + step_size * k3
        k4 = -2*p2_3*p2_3
        p2 += step_size*(k1 + 2*k2 + 2*k3 + k4)/6.0

        p_values[:,index+1] = [p1,1.0-p1,p2,1.0-p2]

    return t_mesh, p_values

def equilibrium_metric(p,p_expected): # RMS Error
    metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
    return np.sqrt(metric_squared)

#%% dt or T Comparison

x1 = .5
x2 = .5

p_actual = np.array([0,1,0,1])
T_array = np.logspace(1, 3, 100)
error_values = np.zeros((3,len(T_array))) 

for index in tqdm(range(len(T_array)),bar_format="{bar:30} {n_fmt}/{total_fmt}"):  
    T = T_array[index]
    
    # Constant dt of .1
    t,solution_matrix = quick_bnn_solver([x1,1-x1],[x2,1-x2],.1,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    error_values[0,index] = equilibrium_metric(p,p_actual)
    
    # Constant dt of 3
    t,solution_matrix = quick_bnn_solver([x1,1-x1],[x2,1-x2],3,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    error_values[1,index] = equilibrium_metric(p,p_actual)
    
    
# Errors
plt.figure(figsize=(10, 6))
plt.plot(np.log10(T_array), np.log10(error_values[0,:]),color="red",label="Constant $\Delta t = .1$")
plt.plot(np.log10(T_array), np.log10(error_values[1,:]),color="green",linestyle='--',label="Constant $\Delta t = 3$")
plt.xlabel("$log_{10}(T)$")
plt.ylabel("$log_{10}$(Equilibrium Error)")
plt.title("BNN Error Order")
plt.legend()
plt.grid(True)
plt.show()

print(error_values[1,:])