import math
from core.point3d import Point3D

class Line:
    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], Point3D) and isinstance(args[1], Point3D):
            p0, p1 = args
            self.x0, self.y0, self.z0 = p0.x, p0.y, p0.z
            self.a = p1.x - p0.x
            self.b = p1.y - p0.y
            self.c = p1.z - p0.z
        elif len(args) == 2 and isinstance(args[0], 'Triangle') and isinstance(args[1], Point3D):
            tri, p0 = args
            self.x0, self.y0, self.z0 = p0.x, p0.y, p0.z
            self.a = tri.na
            self.b = tri.nb
            self.c = tri.nc
        else:
            raise ValueError('Invalid arguments')

    @staticmethod
    def get_alpha(l0, l1):
        return abs(l0.a - l1.a) + abs(l0.b - l1.b) + abs(l0.c - l1.c)

    @staticmethod
    def angle_between(l0, l1):
        dot = l0.a * l1.a + l0.b * l1.b + l0.c * l1.c
        mag0 = math.sqrt(l0.a**2 + l0.b**2 + l0.c**2)
        mag1 = math.sqrt(l1.a**2 + l1.b**2 + l1.c**2)
        if mag0 == 0 or mag1 == 0:
            return 0
        cos_ang = max(-1, min(1, dot / (mag0 * mag1)))
        return math.acos(cos_ang)
