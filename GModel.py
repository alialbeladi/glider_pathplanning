import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import tplquad
from scipy.integrate import dblquad
import RRt_Glider
# from RRt_Glider import *

# Constants
g = 9.81  # gravity acceleration in m/s^2
rho = 1000  # density of water in kg/m^3
cylender_dim = [0.1,0.4] #[0.23,1.22] # [diameter,length]
syringe_dim = [0.05,0.2]
Vol = cylender_dim[1]*np.pi*(cylender_dim[0]/2)**2 # volume for buoyancy calculation in m^3


# Initial poses of masses in the body frame
# Pose matrix includes rotation and position
poses = {
    'a': np.eye(4),  # pose of added water mass
    's': np.eye(4),  # pose of static mass
    'o': np.eye(4),  # pose of movable mass
    'L': np.eye(4),  # pose of the left wing
    'R': np.eye(4),  # pose of the right wing
}
poses['s'][:3, 3] = np.array([0, 0, .1])
poses['L'][:3, 3] = np.array([-.25, 0, 0])
poses['R'][:3, 3] = np.array([+.25, 0, 0])


# Define cross-sectional areas (A) for each part
areas = {
    'L': np.array([0, 0.4*0.2, 0]),#np.array([8.75, 70, 20]),  # example cross-sectional areas for left wing
    'R': np.array([0, 0.4*0.2, 0]),  # example cross-sectional areas for right wing
    'b': np.array([cylender_dim[1]*cylender_dim[0], cylender_dim[1]*cylender_dim[0], np.pi*(cylender_dim[0]/2)**2]),  # example cross-sectional areas for body
}

# Define drag coefficients (c) for each part
drag_coeffs = {
    'L': np.array([1, 0.7, 1.0]),  # example drag coefficients for left wing
    'R': np.array([1.0, 0.7, 1.0]),  # example drag coefficients for right wing
    'b': np.array([0.8, 0.8, 0.8]),  # example drag coefficients for body
}

# Define masses for each point
masses = {
    's': 2,  # mass of the static point
    'o': 0.95#1.#33,  # mass of the movable point
}


def spatial_inertia_matrix_cylinder(Mass, radius, height):
    # Calculate the moments of inertia for the cylinder
    I_xx = I_yy = (1/12) * Mass * (3 * radius**2 + height**2)
    I_zz = (1/2) * Mass * (radius**2)

    # Construct the inertia matrix
    I = np.array([[I_xx, 0, 0],
                  [0, I_yy, 0],
                  [0, 0, I_zz]])

    # Construct the spatial inertia matrix
    G = np.zeros((6, 6))
    G[:3, :3] = I
    G[3:, 3:] = Mass * np.eye(3)

    return G

# Define the spatial inertia matrix (G)
Mass = sum(masses.values())
radius = cylender_dim[0]/2 # in m
height = cylender_dim[1] # in m
G = spatial_inertia_matrix_cylinder(Mass, radius, height)

'''
# Initial conditions
T0 = np.eye(4)
T0_vec = np.hstack((T0[:3, :3].flatten(), T0[:3, 3]))  # flatten initial transformation matrix
V0 = [0,0,0,0,0,0]  # initial twist
S0 = np.hstack((T0_vec, V0))  # initial state vector
'''

t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)  # evaluation points
dt = t_eval[1]
print(dt)

 
 # Define the skew-symmetric matrix for angular velocity
def skew_symmetric(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

 # Define the twist matrix
def twist_matrix(V):
    omega = V[:3]  # angular velocity part
    v = V[3:]      # linear velocity part
    twist = np.zeros((4, 4))
    twist[:3, :3] = skew_symmetric(omega)  # skew-symmetric part for rotation
    twist[:3, 3] = v                       # linear velocity part
    return twist

 # Define the Adjoint matrix
def Adjoint_matrix(R, p):
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew_symmetric(p) @ R
    return Ad

def adjoint_matrix(V):
    omega = V[:3]  # angular velocity part
    v = V[3:]      # linear velocity part
    w_skew = skew_symmetric(omega)
    v_skew = skew_symmetric(v)

    ad_v = np.zeros((6, 6))
    ad_v[:3, :3] = w_skew
    ad_v[3:, 3:] = w_skew
    ad_v[3:, :3] = v_skew
    return ad_v


 # Define the total gravity wrench in the body frame
def total_gravity_wrench(T, poses, masses):
    R_E_b = T[:3, :3]  # Rotation from body to inertial frame
    p_E_b = T[:3, 3]   # Position of the body in the inertial frame

    F_B = np.hstack((np.zeros(3), np.array([0, 0, rho * g * Vol])))
    for key, pose in poses.items():
        if key in ['L', 'R']:
            continue  # Skip L and R for gravity calculation

        # Get mass for the current point
        # print(key)
        m = masses.get(key)

        # Calculate R_iE and p_iE
        R_iB = pose[:3, :3]
        p_iB = pose[:3, 3]
        R_iE = R_iB @ R_E_b.T
        p_iE = p_E_b + R_E_b @ p_iB

        # Calculate f^i_g
        # print(m)
        # print(g)
        f_i_g = R_iE @ np.array([0, 0, -m * g])

        # Calculate the wrench at each point
        F_i_g = np.hstack((np.zeros(3), f_i_g))  # wrench vector

        # Calculate the adjoint matrix
        Ad_T_iB = Adjoint_matrix(R_iB, p_iB)

        # Transform the wrench to the body frame
        F_B_ig = Ad_T_iB.T @ F_i_g
        F_B += F_B_ig

    return F_B

 # Define the drag force
def drag_force(v, A, c):
    # Compute the norm of v
    v_norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    drag = 0.5 * rho * np.diag(c * A * v_norm) @ v
    return (-drag)

def drag_moment_wings(a_y, V_i, A, c):
    # Compute the terms for the matrix elements
    c_z = c[2]
    A_z = A[2]
    w_i = V_i[:3]
    w_x = w_i[0]
    w_y = w_i[1]
    if w_y != 0:
         m = w_x / w_y
    else: m = 0
    term1 = (4/15) * (a_y/2)**5 * (1/2 * rho * c_z * A_z) * (m)
    term2 = (3 * w_x**2 + w_y**2 * (m)**2)
    term3 = (8/15) * w_y * (a_y/2)**5 * (1/2 * rho * c_z * A_z) * w_x * (m)**3

    # Construct the matrix m_D
    m_D = np.array([
        term1 * term2,
        term3,
        0
    ])

    return m_D


def drag_moment_body(w_b, areas, drag_coeffs):
    # define the constants:
    rho = 1000
    pi = np.pi
    L = cylender_dim[1]
    R = cylender_dim[0]/2
    c_x = drag_coeffs[0]
    c_y = drag_coeffs[1]
    c_z = drag_coeffs[2]
    A_x = areas[0]
    A_y = areas[1]
    A_z = areas[2]
    w_x = w_b[0]
    w_y = w_b[1]
    w_z = w_b[2]

    m_D = integral(rho, c_x, A_x, c_y, A_y, c_z, A_z, w_x, w_y, w_z, L, R)
    return m_D

def integral(rho, c_x, A_x, c_y, A_y, c_z, A_z, w_x, w_y, w_z, L, R):
    # Define the integrand functions
    def integrand1(theta, z, rho, c_x, c_y, c_z, w_x, w_y, w_z, A_x, A_y, A_z, L, R):
        v_x = -w_z * R * np.sin(theta) + w_y * z
        v_y = w_z * R * np.cos(theta) - w_x * z
        v_z = -w_y * R * np.cos(theta) + w_x * R * np.sin(theta)
        
        v_norm = np.sqrt(v_x**2 + v_y**2 + v_z**2)
        
        return 0.5 * rho * v_norm * (A_y * c_y * z * (w_z * R * np.cos(theta) - w_x * z) 
                                     - A_z * c_z * R * np.sin(theta) * (-w_y * R * np.cos(theta) 
                                     + w_x * R * np.sin(theta)))

    def integrand2(theta, z, rho, c_x, c_y, c_z, w_x, w_y, w_z, A_x, A_y, A_z, L, R):
        v_x = -w_z * R * np.sin(theta) + w_y * z
        v_y = w_z * R * np.cos(theta) - w_x * z
        v_z = -w_y * R * np.cos(theta) + w_x * R * np.sin(theta)
        
        v_norm = np.sqrt(v_x**2 + v_y**2 + v_z**2)
        
        return 0.5 * rho * v_norm * (A_x * c_x * z * (-w_z * R * np.sin(theta) + w_y * z) 
                                     + A_z * c_z * R * np.cos(theta) * (-w_y * R * np.cos(theta) 
                                     + w_x * R * np.sin(theta)))

    def integrand3(theta, z, rho, c_x, c_y, c_z, w_x, w_y, w_z, A_x, A_y, A_z, L, R):
        v_x = -w_z * R * np.sin(theta) + w_y * z
        v_y = w_z * R * np.cos(theta) - w_x * z
        v_z = -w_y * R * np.cos(theta) + w_x * R * np.sin(theta)
        
        v_norm = np.sqrt(v_x**2 + v_y**2 + v_z**2)
        
        return 0.5 * rho * v_norm * (A_y * c_y * R * np.sin(theta) * (w_z * R * np.cos(theta) - w_x * z) 
                                     - A_x * c_x * R * np.cos(theta) * (-w_z * R * np.sin(theta) 
                                     + w_y * z))

    # Integration limits
    z_min, z_max = -L/2, L/2
    theta_min, theta_max = 0, 2 * np.pi

    # Perform the integration
    I1, _ = dblquad(integrand1, theta_min, theta_max, lambda theta: z_min, lambda theta: z_max, 
                    args=(rho, c_x, c_y, c_z, w_x, w_y, w_z, A_x, A_y, A_z, L, R))
    I2, _ = dblquad(integrand2, theta_min, theta_max, lambda theta: z_min, lambda theta: z_max, 
                    args=(rho, c_x, c_y, c_z, w_x, w_y, w_z, A_x, A_y, A_z, L, R))
    I3, _ = dblquad(integrand3, theta_min, theta_max, lambda theta: z_min, lambda theta: z_max, 
                    args=(rho, c_x, c_y, c_z, w_x, w_y, w_z, A_x, A_y, A_z, L, R))

    # Return the results as a vector
    return np.array([I1, I2, I3])

def drag_moment_body_simple(w_b, C):
    return - np.linalg.norm(w_b)*C @ w_b


 # Define the total drag wrench in the body frame
def total_drag_wrench(V, poses, areas, drag_coeffs):
    A_b = areas['b']
    c_b = drag_coeffs['b']
    v_b = V[3:]
    w_b = V[:3]
    f_D_b = drag_force(v_b, A_b, c_b)
    m_D_b1 = np.zeros(3)
    # if poses['o'][:3, 3][0] == 0: m_D_b = drag_moment_body(w_b, A_b, c_b)
    m_D_b = drag_moment_body_simple(w_b, np.diag([0.2,0.2,0.1]))

    F_H = np.hstack((m_D_b, f_D_b))

    for key in ['L', 'R']:  # Calculate drag for the left wing, right wing, and body
        pose = poses[key]
        A = areas[key]
        c = drag_coeffs[key]

        # Calculate R_iB and p_iB
        R_iB = pose[:3, :3]
        p_iB = pose[:3, 3]

        # Transform the twist from the body frame to the part frame
        Ad_T_iB = Adjoint_matrix(R_iB, p_iB)
        V_i = Ad_T_iB @ V

        # Extract linear velocity part of the twist
        v_i = V_i[3:]

        # Calculate drag force in the part frame
        f_i_D = drag_force(v_i, A, c)

        # Compute the moment
        a_y = A[1]
        m_i_D = drag_moment_wings(a_y , V_i, A, c)

        # Combine drag force and moment in the part frame
        F_i_D = np.hstack((m_i_D, f_i_D))

        # Transform the drag wrench to the body frame
        F_B_iD = Ad_T_iB.T @ F_i_D
        F_H += F_B_iD

    return F_H

 # Define the system of ODEs
def system_odes(t, S, G, poses, masses, areas, drag_coeffs, u1_a, u1_b, u2):
    poses['o'][:3, 3] = np.array([u1_a, u1_b, 0])
    poses['a'][:3, 3] = np.array([0, 0, u2/2])
    masses['a'] = rho * np.pi * (syringe_dim[0]/2)**2 * u2

    T = np.eye(4)
    T[:3, :3] = S[:9].reshape((3, 3))  # extract rotation part
    T[:3, 3] = S[9:12]  # extract position part
    V = S[12:]  # extract twist
    
    dTdt = T @ twist_matrix(V)  # compute the time derivative of the transformation matrix

    adv = adjoint_matrix(V)  # compute the adjoint matrix
    F_gravity = total_gravity_wrench(T, poses, masses)  # compute the total gravity wrench
    F_drag = total_drag_wrench(V, poses, areas, drag_coeffs)  # compute the total drag wrench
    F = F_gravity + F_drag  # total external force
    dVdt = np.linalg.inv(G) @ (F + adv.T @ G @ V)  # compute the time derivative of the twist

    dSdt = np.hstack((dTdt[:3, :3].flatten(), dTdt[:3, 3], dVdt))  # combine results

     # Debugging: print intermediate results

    # print(f"time: {t}")
    """
    print(f"T: \n{T}")
    print(f"V: \n{V}")
    print(f"dTdt: \n{dTdt}")
    print(f"F_gravity: \n{F_gravity}")
    print(f"F_drag: \n{F_drag}")
    print(f"F: \n{F}")
    print(f"dVdt: \n{dVdt}")
    print(f"dSdt: \n{dSdt}")
    """
    

    return dSdt*dt






