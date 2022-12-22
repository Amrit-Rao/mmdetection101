from pydantic import BaseModel
import math


class Complex(BaseModel):
    x: float
    y: float
    epsilon = 1e-6

    def __add__(self, n):
        return Complex(x=self.x + n.x, y=self.y + n.y)

    def __sub__(self, n):
        return Complex(x=self.x - n.x, y=self.y - n.y)

    def __mul__(self, n):
        # (a+ib)(x+iy) = (ax - by) + i(bx + ay)
        return Complex(x=self.x * n.x - self.y * n.y, y=self.y * n.x + self.x * n.y)

    def __truediv__(self, n):
        # (a+ib)/(x+iy) = (a+ib)*(x-iy)/(x^2+y^2)
        mod_n = n.x**2 + n.y**2
        conj_n = Complex(x=n.x, y=-n.y)
        numerator = self * conj_n
        try:
            return Complex(x=numerator.x / mod_n, y=numerator.y / mod_n)
        except ZeroDivisionError:
            raise ZeroDivisionError("The denominator is zero")

    def __pow__(self, n):
        # a^b = (re^(i0))^(x+iy)
        r = math.sqrt(self.x**2 + self.y**2)
        try:
            theta = math.atan(self.y / self.x)
        except ZeroDivisionError:
            if self.y > 0:
                theta = math.pi / 2
            else:
                theta = -math.pi / 2
        # final r = r^x*e^(-y0)
        final_r = (r**n.x) * math.e ** (-theta * n.y)
        # e^(i0x)
        term_1 = Complex(x=math.cos(theta * n.x), y=math.sin(theta * n.x))
        # r^(iy)
        term_2 = Complex(x=math.cos(n.y * math.log(r)), y=math.sin(n.y * math.log(r)))
        # Multiply e^(i0) and r^(iy)
        final_term = term_1 * term_2
        return Complex(x=final_term.x * final_r, y=final_term.y * final_r)

    def __eq__(self, n):
        return (abs(self.x - n.x) < self.epsilon) and (abs(self.y - n.y) < self.epsilon)
