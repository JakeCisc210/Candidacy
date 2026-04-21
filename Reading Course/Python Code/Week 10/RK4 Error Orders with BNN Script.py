import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% New ODE Solver System

def quick_bnn_solver(p1, p2, h, T):
    p10 = float(p1[0])
    p20 = float(p2[0])

    N = int(np.ceil(T/h)) 
    t_mesh = np.linspace(0,T,N+1)
    step_size = T/N

    p_values = np.zeros((4, N+1), dtype=float)
    p_values[:, 0] = [p10,1-p10,p20,1-p20]

    p1 = p10
    p2 = p20

    for index in range(N):
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

    return t_mesh, p_values, step_size

def equilibrium_metric(p,p_expected): # RMS Error
    metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
    return np.sqrt(metric_squared)

#%% Error Order

p1 = [1/2,1/2]
p2 = [1/2,1/2]
T = 10
h_array = np.logspace(-1, -3, 100)

# Exact solutions
x1 = 1/(2*T+2)
y1 = (2*T+1)/(2*T+2)
x2 = 1/(2*T+2)
y2 = (2*T+1)/(2*T+2)

metric_array = np.zeros((len(h_array)))
dt_array = np.zeros((len(h_array)))

counter = 0
for h in tqdm(h_array,bar_format="{bar:30} {n_fmt}/{total_fmt}"): 
    t,solution_matrix,h_real = quick_bnn_solver(p1,p2,h,T)
    p = np.array([solution_matrix[0,-1],solution_matrix[1,-1],solution_matrix[2,-1],solution_matrix[3,-1]])
    p_actual = np.array([x1, y1, x2, y2])
    metric_array[counter] = equilibrium_metric(p,p_actual)
    dt_array[counter] = h_real
    counter += 1

# Errors
plt.figure(figsize=(10, 6))
plt.plot(np.log10(dt_array), np.log10(metric_array),color="red",label="Error")
plt.plot(np.log10(dt_array), -3.75+4*np.log10(dt_array),label='O($h^{-4}$)', color='black',linestyle='--',)
plt.legend()
plt.xlabel("log10(Step Size)")
plt.ylabel("log10(Error")
plt.title("BNN Error Order")
plt.grid(True)
plt.show()