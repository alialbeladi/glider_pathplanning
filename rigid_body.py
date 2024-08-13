from geometry import Pose, Vector
import numpy as np


# class Force(Vector):
#     def __init__(self,x=0,y=0,z=0, vect=None):
#         super().__init__(x,y,z,vect)
#
#
# class Moment(Vector):
#     def __init__(self,x=0,y=0,z=0, vect=None):
#         super().__init__(x,y,z,vect)


class Wrench:
    def __init__(self, force=Vector(), moment=Vector(), vect=None):
        if vect is None:
            self.force = force
            self.moment = moment
        else:
            self.from_vect(vect)

    def from_vect(self, vect=np.zeros(6)):
        self.force = Vector(vect=vect[3:])
        self.moment = Vector(vect=vect[:3])

    def as_vect(self):
        return np.hstack([self.moment.as_vect(),self.force.as_vect()])

    def transform(self, pose=Pose()):
        F = pose.inverse().Ad().T @ self.as_vect()
        transformed = Wrench(F)
        return transformed


class Body:
    def __init__(self, pose=Pose(), mass_matrix=np.eye(6)):
        self.pose = pose
        self.mass_matrix = mass_matrix