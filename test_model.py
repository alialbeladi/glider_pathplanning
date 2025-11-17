import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
from GModel import system_odes, G, poses, masses, areas, drag_coeffs, radius,syringe_dim

T0 = np.eye(4)
V0 = np.zeros(6)
S0 = np.hstack((T0[:3, :3].flatten(), T0[:3, 3], V0))
print(S0)
u = np.array([0.05,0.05,0.08])
sol = solve_ivp(system_odes, [0, 20], S0,
                                args=(G, poses, masses, areas, drag_coeffs, u[0], u[1], u[2]), rtol=1e-6, atol=1e-8)

path_x = sol.y[9, :]
path_y = sol.y[10, :]
path_z = sol.y[11, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(path_x, path_y, path_z, "-g")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.grid(True)
plt.show()