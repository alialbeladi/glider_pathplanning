import math
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from GModel import *

show_animation = True



class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, T, V):  
            self.T = T
            self.V = V
            self.path_x = []
            self.path_y = []
            self.path_z = []

            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])
            self.zmin = float(area[4]) 
            self.zmax = float(area[5]) 

    def __init__(self,
                 
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis= 1.0,
                 path_resolution= 0.5,
                 goal_sample_rate= 5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 
                 ):
        """
        Setting Parameter

        start:Start Position [T]
        goal:Goal Position [T]
        obstacleList:obstacle Positions [[x,y,z,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax,zmin,zmax]
        robot_radius: robot body modeled as circle with given radius

        """
        
        start_T = np.eye(4)
        start_T[:3, 3] = np.array([start[0], start[1], start[2]])
        self.start = self.Node(start_T,V = [0,0,0,0,0,0])
        end_T = np.eye(4)
        end_T[:3, 3] = np.array([goal[0], goal[1], goal[2]])
        self.end = self.Node(end_T,V = [0,0,0,0,0,0]) 
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(
                   new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(ax, rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].T[:3, 3][0],
                                      self.node_list[-1].T[:3, 3][1],
                                      self.node_list[-1].T[:3, 3][2]) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(
                        final_node, self.obstacle_list, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(ax, rnd_node)

        plt.show()
        return None  # cannot find path
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        
        new_node = self.Node(from_node.T, from_node.V)
        positionTo = to_node.T[:3, 3]

        T0 = from_node.T
        V0 = np.array(from_node.V).flatten()
        T0_vec = np.hstack((T0[:3, :3].flatten(), T0[:3, 3]))  # flatten initial transformation matrix
        S0 = np.hstack((T0_vec, V0))  # initial state vector
        # Define control space U
        u1_a = np.linspace(-radius, radius, num=10)
        u1_b = np.linspace(-radius, radius, num=10)
        u2 = np.linspace(0, 0.6, num=10)

        # Iterate over control space
        best_q = None
        min_dist = float("inf")
        T_new = None
        V_new = None

        # Run 20 iterations
        for _ in range(10):
           # Select random values from u1_a, u1_b, and u2
           ua = random.choice(u1_a)
           ub = random.choice(u1_b)
           u = random.choice(u2)

           # Integrate the dynamics
           sol = solve_ivp(system_odes, t_span, S0, args=(G, poses, masses, areas, drag_coeffs, ua, ub, u), rtol=1e-6, atol=1e-8)

           # Get the final state from integration
           T_sol = sol.y[:12].T
           q_final = T_sol[:3, 3]

           # Calculate distance to target node
           dist_to_target = np.linalg.norm(q_final - np.array([positionTo[0], positionTo[1], positionTo[2]]))

           # Check if this is the best solution so far
           if dist_to_target < min_dist:
              min_dist = dist_to_target
              best_q = q_final
              T_new = T_sol 
              V_new = sol.y[12:].T
        
        '''
        for ua in u1_a:
          for ub in u1_b:
            for u in u2:
                # Integrate the dynamics
                sol = solve_ivp(system_odes,t_span, S0, args=(G, poses, masses, areas, drag_coeffs, ua, ub, u), rtol=1e-6, atol=1e-8)

                # Get the final state from integration
                T_sol = sol.y[:12].T
                q_final = T_sol[:3, 3]
                
                # Calculate distance to target node
                dist_to_target = np.linalg.norm(q_final - np.array([positionTo[0], positionTo[1], positionTo[2]]))
                
                if dist_to_target < min_dist:
                    min_dist = dist_to_target
                    best_q = q_final
                    T_new = T_sol 
                    V_new = sol.y[12:].T
    
        '''

    
        # Update new_node with the best state found
        if best_q is not None:
         new_node.T = T_new
         new_node.V = V_new
         new_node.path_x.append(best_q[0])
         new_node.path_y.append(best_q[1])
         new_node.path_z.append(best_q[2])
       
        new_node.parent = from_node

        return new_node


    def generate_final_course(self, goal_ind):
        path = self.end.T[:3, 3]
        node = self.node_list[goal_ind]
        while node.parent is not None:

            path.append(node.T[:3, 3])
            node = node.parent
        path.append(node.T[:3, 3])

        return path

    def calc_dist_to_goal(self, x, y, z):
        dx = x - self.end.T[:3, 3][0]
        dy = y - self.end.T[:3, 3][1]
        dz = z - self.end.T[:3, 3][2]
        return math.hypot(dx, dy, dz)

    def get_random_node(self):
        rand_T = np.eye(4)
        if random.randint(0, 100) > self.goal_sample_rate:
            x = random.uniform(self.min_rand, self.max_rand)
            y = random.uniform(self.min_rand, self.max_rand)
            z = random.uniform(self.min_rand, self.max_rand)
            rand_T[:3, 3] = np.array([x,y,z])
            rnd = self.Node( rand_T, V = [0,0,0,0,0,0]
            )
        else:  # goal point sampling
            rand_T[:3, 3] = self.end.T[:3, 3]
            rnd = self.Node(rand_T, V = [0,0,0,0,0,0])  
        return rnd

    def draw_graph(self, ax, rnd=None):
        plt.cla()
        if rnd is not None:
            ax.scatter(rnd.T[:3, 3][0], rnd.T[:3, 3][1], rnd.T[:3, 3][2], c='k', marker='^')  # 3D scatter plot

        for node in self.node_list:
            if node.parent:
                ax.plot(node.path_x, node.path_y, node.path_z, "-g")  # 3D plot

        for (ox, oy, oz, size) in self.obstacle_list:
            self.plot_sphere(ax, ox, oy, oz, size)  # 3D obstacles

        if self.play_area is not None:
            # Code to plot the play area in 3D
            pass

        ax.scatter(self.start.T[:3, 3][0], self.start.T[:3, 3][1], self.start.T[:3, 3][2], c='r', marker='x')
        ax.scatter(self.end.T[:3, 3][0], self.end.T[:3, 3][1], self.end.T[:3, 3][2], c='r', marker='x')
        ax.set_xlim([self.min_rand, self.max_rand])
        ax.set_ylim([self.min_rand, self.max_rand])
        ax.set_zlim([self.min_rand, self.max_rand])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_sphere(ax, x, y, z, size):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xs = x + size * np.outer(np.cos(u), np.sin(v))
        ys = y + size * np.outer(np.sin(u), np.sin(v))
        zs = z + size * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        
        dlist = [(node.T[:3, 3][0] - rnd_node.T[:3, 3][0])**2 + (node.T[:3, 3][1] - rnd_node.T[:3, 3][1])**2 + (node.T[:3, 3][2] - rnd_node.T[:3, 3][2])**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.T[:3, 3][0] < play_area.xmin or node.T[:3, 3][0] > play_area.xmax or \
           node.T[:3, 3][1] < play_area.ymin or node.T[:3, 3][1] > play_area.ymax or \
           node.T[:3, 3][2] < play_area.zmin or node.T[:3, 3][2] > play_area.zmax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):

        if node is None:
            return False

        for (ox, oy, oz, size) in obstacleList:  # Add z coordinate and size
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            dz_list = [oz - z for z in node.path_z]  # Track z path
            d_list = [dx * dx + dy * dy + dz * dz for (dx, dy, dz) in zip(dx_list, dy_list, dz_list)]  # 3D distance

            if min(d_list) <= (size + robot_radius) ** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z  # Add z dimension
        d = math.sqrt(dx**2 + dy**2 + dz**2)  # 3D distance
        theta = math.atan2(dy, dx)
        phi = math.atan2(math.sqrt(dx**2 + dy**2), dz)  # Angle in 3D
        return d, theta, phi  # Return 3D angles


def main():
    
    print("Start RRT path planning")
    

    # ====Search Path with RRT====
    obstacle_list = []  # [x, y, z, radius]
    # Set Initial parameters
    rrt = RRT(start=[0, 0, 0],
              goal=[10, 10, 10],
              rand_area=[0, 10],
              obstacle_list=obstacle_list,
              play_area=[0, 10, 0, 10, 0, 10]
              )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!!")

        # Draw final path
        if show_animation:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            rrt.draw_graph(ax)
            ax.plot([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], '-r')
            plt.show()
    


if __name__ == '__main__':
    main()
