import math
from core.point3d import Point3D
from core.line import Line
from core.interpolate import quaficient

class Triangle:
    def __init__(self, p0=None, p1=None, p2=None):
        if p0 is None or p1 is None or p2 is None:
            return
        dd = self.get_d(p0, p1)
        aa = [[p0.x, p0.y, p0.z],
              [p1.x, p1.y, p1.z],
              [p2.x, p2.y, p2.z]]
        ddd = [dd.x, dd.y, dd.z]
        cc = quaficient(aa, ddd)
        self.a, self.b, self.c = cc
        self.d = self.a * p0.x + self.b * p0.y + self.c * p0.z
        l0 = Line(p0, p1)
        l1 = Line(p0, p2)
        self.na = (l0.b * l1.c) - (l0.c * l1.b)
        self.nb = (l0.c * l1.a) - (l0.a * l1.c)
        self.nc = (l0.a * l1.b) - (l0.b * l1.a)

    def get_d(self, p0, p1):
        l0 = Line(p0, p1)
        return Point3D(p1.x + l0.a * 2, p1.y + l0.b * 2, p1.z + l0.c * 2)

    # سایر متدهای triangle به‌دلیل طولانی‌شدن در اینجا فشرده نمی‌شوند.
    # می‌توانید نسخهٔ کامل را از کدهای قبلی استخراج کنید.
    # برای API حاضر، بخش reduction و angle collection ضروری نیستند.
