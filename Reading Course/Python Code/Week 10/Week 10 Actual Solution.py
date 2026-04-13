import numpy as np
import matplotlib.pyplot as plt

# Time grid
t = np.linspace(1, 100, 1005)

# Exact solutions
x1 = 1 / (2*t + 2)
y1 = (2*t + 1) / (2*t + 2)
x2 = 1 / (2*t + 2)
y2 = (2*t + 1) / (2*t + 2)

# Plot Trajectories
plt.figure(figsize=(10, 6))
plt.plot(t, x1, label=r"$x^1$", color='red')
plt.plot(t, y1, label=r"$y^1$", color='blue')
plt.plot(t, x2, label=r"$x^2$",linestyle='--', color='yellow')
plt.plot(t, y2, label=r"$y^2$",linestyle='--', color='cyan')

plt.xlabel("Time $t$", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Exact Solutions of the BNN ODE System", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Error Convergence
def equilibrium_metric(p,p_expected): # RMS Error
    metric_squared = np.dot(p-p_expected,p-p_expected)/len(p)
    return np.sqrt(metric_squared)

p_actual = np.array([0, 1, 0, 1])

errors = np.zeros(len(t))
for index in range(len(t)):
    p = np.array([x1[index],y1[index],x2[index],y2[index]])
    errors[index] = equilibrium_metric(p,p_actual)

plt.figure(figsize=(10, 6))
plt.plot(np.log10(t), np.log10(errors),label='error', color='red')
plt.plot(np.log10(t),-.25-np.log10(t),label='O($t^{-1}$)', color='black',linestyle='--',)
plt.xlabel("$log_{10}$($t$)", fontsize=12)
plt.ylabel("$log_{10}$(RMS Error)", fontsize=12)
plt.title("Convergence of RMS Error", fontsize=14)
plt.legend()

