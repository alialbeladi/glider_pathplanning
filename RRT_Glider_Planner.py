import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
from GModel import system_odes, G, poses, masses, areas, drag_coeffs, radius,syringe_dim
show_animation = True

class RRT:
    class Node:
        def __init__(self, T, V,inputs=np.array([0,0,syringe_dim[1]/2])):
            self.T = T  # Transformation matrix
            self.V = V  # Velocity vector
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.inputs = inputs
            self.parent = None


    class AreaBounds:
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])
            self.zmin = float(area[4])
            self.zmax = float(area[5])

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, path_resolution=5,
                 goal_sample_rate=5, max_iter=200, play_area=None, robot_radius=0.0):
        start_T = np.eye(4)
        start_T[:3, 3] = np.array([start[0], start[1], start[2]])
        self.start = self.Node(start_T, V=[0, 0, 0, 0, 0, 0])
        
        end_T = np.eye(4)
        end_T[:3, 3] = np.array([goal[0], goal[1], goal[2]])
        self.end = self.Node(end_T, V=[0, 0, 0, 0, 0, 0])
        
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None

    def planning(self, animation=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        self.node_list = [self.start]
        self.generate_initial_nodes(self.start)
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node) and self.check_collision(new_node):
                print("Append Node")
                self.node_list.append(new_node)

            if animation and i % 50 == 0:
                self.draw_graph(ax, rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1]) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)
                
            if animation and i % 50:
                self.draw_graph(ax, rnd_node)

        plt.show()
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        
        new_node = self.Node(np.copy(from_node.T), from_node.V[:])
        new_node.path_x = [new_node.T[:3, 3][0]]
        new_node.path_y = [new_node.T[:3, 3][1]]
        new_node.path_z = [new_node.T[:3, 3][2]]
        T0 = from_node.T
        V0 = np.array(from_node.V).flatten()
        S0 = np.hstack((T0[:3, :3].flatten(), T0[:3, 3], V0))

        print(S0[9:12])
        u1_a = np.linspace(-radius/10, radius/10, num=2)
        u1_b = np.linspace(-radius/10, radius/10, num=2)
        u2 = np.linspace(-syringe_dim[1]/10, syringe_dim[1]/10, num=3)

        best_q = None
        min_dist = float("inf")

        for i,du_i in enumerate([u1_a,u1_b,u2]):
            for j in range(len(du_i)):
                # ua = random.choice(u1_a)
                # ub = random.choice(u1_b)
                # u = random.choice(u2)
                du = np.array([0.0,0.0,0.0])
                du[i]=du_i[j]
                u = from_node.inputs + du
                # Check if inputs exceed limits:
                u[:2][u[:2]>radius] = radius
                u[:2][u[:2]<-radius]=-radius
                u[2] = syringe_dim[1] if u[2]>syringe_dim[1] else u[2]
                u[2] = 0 if u[2] < 0 else u[2]
                # print(u)
                sol = solve_ivp(system_odes, [0, self.path_resolution], S0, args=(G, poses, masses, areas, drag_coeffs, u[0], u[1], u[2]), rtol=1e-6, atol=1e-8)
                T_sol = sol.y[:12].reshape(-1, 12)
                q_final = T_sol[-1, 9:12]
                # print(q_final)

                dist_to_target = np.linalg.norm(q_final - to_node.T[:3, 3])
                if dist_to_target < min_dist:
                    min_dist = dist_to_target
                    min_q_final = q_final
                    new_node.T = np.copy(T_sol[-1].reshape(3, 4))
                    new_node.V = sol.y[12:, -1]
                    new_node.inputs = u.copy()

        new_node.path_x = sol.y[9,:]    #.append(min_q_final[0])  # x-coordinate
        new_node.path_y = sol.y[10,:]    #.append(min_q_final[1]) # y-coordinate
        new_node.path_z = sol.y[11,:]    #.append(min_q_final[2]) # z-coordinate
        new_node.parent = from_node
        return new_node

    def generate_initial_nodes(self, from_node):

        T0 = from_node.T
        V0 = np.array(from_node.V).flatten()
        S0 = np.hstack((T0[:3, :3].flatten(), T0[:3, 3], V0))

        print(S0[9:12])
        u1_a = np.linspace(-radius / 10, radius / 10, num=2)
        u1_b = np.linspace(-radius / 10, radius / 10, num=2)
        u2 = np.linspace(-syringe_dim[1] / 10, syringe_dim[1] / 10, num=3)



        for i, du_i in enumerate([u1_a, u1_b, u2]):
            for j in range(len(du_i)):
                new_node = self.Node(np.copy(from_node.T), from_node.V[:])
                new_node.path_x = [new_node.T[:3, 3][0]]
                new_node.path_y = [new_node.T[:3, 3][1]]
                new_node.path_z = [new_node.T[:3, 3][2]]
                du = np.array([0.0, 0.0, 0.0])
                du[i] = du_i[j]
                u = from_node.inputs + du
                # Check if inputs exceed limits:
                u[:2][u[:2] > radius] = radius
                u[:2][u[:2] < -radius] = -radius
                u[2] = syringe_dim[1] if u[2] > syringe_dim[1] else u[2]
                u[2] = 0 if u[2] < 0 else u[2]
                print(u)
                sol = solve_ivp(system_odes, [0, self.path_resolution], S0,
                                args=(G, poses, masses, areas, drag_coeffs, u[0], u[1], u[2]), rtol=1e-6, atol=1e-8)
                T_sol = sol.y[:12].reshape(-1, 12)
                q_final = T_sol[-1, 9:12]
                print(q_final)

                new_node.T = np.copy(T_sol[-1].reshape(3, 4))
                new_node.V = sol.y[12:, -1]
                new_node.inputs = u.copy()

                new_node.path_x = sol.y[9, :]  # .append(min_q_final[0])  # x-coordinate
                new_node.path_y = sol.y[10, :]  # .append(min_q_final[1]) # y-coordinate
                new_node.path_z = sol.y[11, :]  # .append(min_q_final[2]) # z-coordinate
                new_node.parent = from_node
                self.node_list.append(new_node)

    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node is not None:
            path.append(node.T[:3, 3])
            node = node.parent
        
        return path

    def calc_dist_to_goal(self, node):
        dx = node.T[:3, 3][0] - self.end.T[:3, 3][0]
        dy = node.T[:3, 3][1] - self.end.T[:3, 3][1]
        dz = node.T[:3, 3][2] - self.end.T[:3, 3][2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_T = np.eye(4)
            rand_T[:3, 3] = np.array([random.uniform(self.min_rand, self.max_rand) for _ in range(3)])
            rnd = self.Node(rand_T, [0, 0, 0, 0, 0, 0])
        else:
            rnd = self.Node(np.copy(self.end.T), [0, 0, 0, 0, 0, 0])
        return rnd

    def draw_graph(self, ax, rnd=None):
        plt.cla()
        ax.scatter(self.start.T[:3, 3][0], self.start.T[:3, 3][1], self.start.T[:3, 3][2], c='r', marker='x')
        ax.scatter(self.end.T[:3, 3][0], self.end.T[:3, 3][1], self.end.T[:3, 3][2], c='r', marker='x')
        if rnd:
            ax.scatter(rnd.T[:3, 3][0], rnd.T[:3, 3][1], rnd.T[:3, 3][2], c='k', marker='*')
        for node in self.node_list:
            if node.parent:
                print(node.T[:3, 3])
                # ax.plot(node.T[:3, 3], node.T[:3, 3][1], node.T[:3, 3][2], "-g")
                ax.plot(node.path_x, node.path_y, node.path_z, "-g")
        for (ox, oy, oz, size) in self.obstacle_list:
            self.plot_sphere(ax, ox, oy, oz, size)
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

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(np.linalg.norm(node.T[:3, 3] - rnd_node.T[:3, 3])) for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def check_if_outside_play_area(self, node):
        if self.play_area is None:
            return True
        if node.T[:3, 3][0] < self.play_area.xmin or node.T[:3, 3][0] > self.play_area.xmax or \
           node.T[:3, 3][1] < self.play_area.ymin or node.T[:3, 3][1] > self.play_area.ymax or \
           node.T[:3, 3][2] < self.play_area.zmin or node.T[:3, 3][2] > self.play_area.zmax:
            return False
        return True

    def check_collision(self, node):
        for (ox, oy, oz, size) in self.obstacle_list:
            dx = node.T[:3, 3][0] - ox
            dy = node.T[:3, 3][1] - oy
            dz = node.T[:3, 3][2] - oz
            d = dx**2 + dy**2 + dz**2
            if d <= (size + self.robot_radius) ** 2:
                return False
        return True
def main():
    
    print("Start RRT path planning")
    

    # ====Search Path with RRT====
    obstacle_list = []  # [x, y, z, radius]
    # Set Initial parameters
    rrt = RRT(start=[2, 2, 2],
              goal=[5, 5, 3],
              rand_area=[0, 6],
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
