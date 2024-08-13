import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List

class Vector:
    def __init__(self, x=0,y=0,z=0, vect=None):
        if vect is None:
            self.x = x
            self.y = y
            self.z = z
        else:
            self.from_vect(vect)

    def as_vect(self):
        return np.array([self.x,self.y,self.z])

    def from_vect(self,vect = np.zeros((3))):
        self.x = vect[0]
        self.y = vect[1]
        self.z = vect[2]

    def skew_symmetric(self):
        return np.array([
            [0, -self.z, self.y],
            [self.z, 0, -self.x],
            [-self.y, self.x, 0]])

    def norm(self):
        return np.linalg.norm(self.as_vect())

class Orientation(R):
    def __init__(self,w=0,x=0,y=0,z=1,rotM=None,axang=None):
        if rotM is not None:
            q = R.from_matrix(rotM)
            super().__init__(q.as_quat())
        elif axang is not None:
            q = R.from_rotvec(axang)
            super().__init__(q.as_quat())
        else:
            super().__init__(quat=np.array([w,x,y,z]))
    def ex(self):
        return self.as_matrix()[:3,0]
    def ey(self):
        return self.as_matrix()[:3,1]
    def ez(self):
        return self.as_matrix()[:3,2]

class Pose:
    def __init__(self,position=Vector(), orientation=Orientation(),se3 = None, frame=''):
        self.frame = frame
        self.position = position
        self.orientation = orientation
        if se3 is not None:
            self.from_SE3(se3)

    def as_SE3(self):
        X = np.eye(4)
        X[:3,:3] = self.orientation.as_matrix().copy()
        X[:3,3] = self.position.as_vect()
        return X

    def from_SE3(self,X = np.eye(4)):
        self.position = Vector(vect = X[:3,3])
        self.orientation = Orientation(rotM=X[:3,:3])

    def mult(self,p2=None):
        X = self.as_SE3()
        p = Pose()
        p.from_SE3(X@p2.as_SE3())
        return p

    def inverse(self):
        X_inv = np.linalg.inv(self.as_SE3())
        p = Pose()
        p.from_SE3(X_inv)
        return p

    def Ad(self):
        R = self.orientation.as_matrix().copy()
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R
        Ad[3:, 3:] = R
        Ad[3:, :3] = self.position.skew_symmetric() @ R
        return Ad

class Poses:
    def __init__(self, poses: List[Pose]):
        self.poses = {}
        self.initialize_poses(poses)
        
    def initialize_poses(self, poses):
        for pose in poses:
            if len(pose.frame)==1:
                self.poses[pose.frame] = pose
        for pose in poses:
            if len(pose.frame) == 2:
                if pose.frame[0] in self.poses.keys():
                    pose_parent = self.poses.pop(pose.frame[0])
                    self.poses[pose.frame[1]] = Pose(se3= pose_parent @ pose.as_SE3())
                elif pose.frame[1] in self.poses.keys():
                    pose_parent = self.poses.pop(pose.frame[1])
                    self.poses[pose.frame[0]] = Pose(se3= pose_parent @ pose.inverse().as_SE3())

    def find_pose(self,frame):
        self.poses[0].frame

class Twist:
    def __init__(self, linear=Vector(), angular=Vector(), vect = None, frame=''):
        self.frame = frame
        if vect is None:
            self.linear = linear
            self.angular = angular
        else:
            self.from_vect(vect)

    def from_vect(self, vect=np.zeros(6)):
        self.linear = Vector(vect=vect[3:])
        self.angular = Vector(vect=vect[:3])

    def twist_matrix(self):
        twist = np.zeros((4, 4))
        twist[:3, :3] = self.angular.skew_symmetric()# skew-symmetric part for rotation
        twist[:3, 3] = self.linear.as_vect()  # linear velocity part
        return twist

    def ad(self):
        ad_v = np.zeros((6, 6))
        ad_v[:3, :3] = self.angular.skew_symmetric()
        ad_v[3:, 3:] = self.angular.skew_symmetric()
        ad_v[3:, :3] = self.linear.skew_symmetric()
        return ad_v


if __name__=="__main__":
    p1 = Pose(orientation=Orientation(axang=-np.pi/20*np.array([0,1,0])),position=Vector(vect=np.array([1,0,-2])))
    p2 = Pose(orientation=Orientation(axang=-np.pi/2*np.array([0,1,0])),position=Vector(vect=np.array([1,2,3])))

    print(p1.as_SE3())
    print(p2.as_SE3())

    p3 = p1.mult(p2)

    print(p1.as_SE3())
    print(p2.as_SE3())

    A = np.zeros((3,3))
    d = np.zeros((3,))
    A[:,0] = p1.orientation.ez()
    A[:,1:] = p2.orientation.as_matrix()[:,:2]
    d[0] = A[:,0].T @ p1.position.as_vect()
    d[1:] = A[:,1:].T @ p2.position.as_vect()
    print(A,d,)
    intersection = np.linalg.inv(A.T)@d
    print(intersection)
    X1inv = np.linalg.inv(p1.as_SE3())
    X2inv = np.linalg.inv(p2.as_SE3())
    print(X1inv[:3,:3]@intersection + X1inv[:3,3])
    print(X2inv[:3, :3] @ intersection + X2inv[:3, 3])

    print(R.from_matrix(np.eye(3)).as_quat())
    print(Orientation(0,0,0,1).as_matrix())
