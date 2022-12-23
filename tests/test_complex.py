from segmentation import Complex
import pytest


class TestAdd:
    def test_one(self):
        x = Complex(x=1, y=4)
        y = Complex(x=3, y=-8)
        assert x + y == Complex(x=4, y=-4)

    def test_two(self):
        x = Complex(x=98457635648, y=-2523449274358385687432)
        y = Complex(x=67813493425, y=76273598073625)
        assert x + y == Complex(x=166271129073, y=-2523449198084787613807)

    def test_three(self):
        x = Complex(x=0, y=1)
        y = Complex(x=1, y=-0)
        assert x + y == Complex(x=1, y=1)

    def test_four(self):
        x = Complex(x=19108543.15681763415, y=0.91973859035642)
        y = Complex(x=-314598817495.9188910355, y=34256.413254268910254)
        assert x + y == Complex(x=-3.1457970895276207340135e11, y=34257.332992859266674)

    def test_five(self):
        x = Complex(x=6, y=8)
        y = Complex(x=-6, y=-8)
        assert x + y == Complex(x=0, y=0)

    def test_six(self):
        x = Complex(
            x=922548813.4133333389547892, y=9147385763256475.1245354981743561754
        )
        y = Complex(x=0, y=0)
        assert x + y == Complex(
            x=922548813.4133333389547892, y=9147385763256475.1245354981743561754
        )


class TestSub:
    def test_one(self):
        x = Complex(x=1, y=4)
        y = Complex(x=3, y=-8)
        assert x - y == Complex(x=-2, y=12)

    def test_two(self):
        x = Complex(x=98457635648, y=-2523449274358385687432)
        y = Complex(x=67813493425, y=76273598073625)
        assert x - y == Complex(x=30644142223, y=-2523449350631983761057)

    def test_three(self):
        x = Complex(x=0, y=1)
        y = Complex(x=1, y=-0)
        assert x - y == Complex(x=-1, y=1)

    def test_four(self):
        x = Complex(x=19108543.15681763415, y=0.91973859035642)
        y = Complex(x=-314598817495.9188910355, y=34256.413254268910254)
        assert x - y == Complex(x=3.1461792603907570866965e11, y=-34255.49351567856)

    def test_five(self):
        x = Complex(x=6, y=8)
        y = Complex(x=6, y=8)
        assert x - y == Complex(x=0, y=0)

    def test_six(self):
        x = Complex(
            x=922548813.4133333389547892, y=9147385763256475.1245354981743561754
        )
        y = Complex(x=0, y=0)
        assert x - y == Complex(
            x=922548813.4133333389547892, y=9147385763256475.1245354981743561754
        )


class TestMul:
    def test_one(self):
        x = Complex(x=1, y=4)
        y = Complex(x=3, y=-8)
        assert x * y == Complex(x=35, y=4)

    def test_two(self):
        x = Complex(x=98457635648, y=-2523449274358385687432)
        y = Complex(x=67813493425, y=76273598073625)
        assert x * y == Complex(
            x=192472555711598847433005823266795400, y=-171123903265305280213319108550600
        )

    def test_three(self):
        x = Complex(x=0, y=1)
        y = Complex(x=1, y=-0)
        assert x * y == Complex(x=0, y=1)

    def test_four(self):
        x = Complex(x=19108543.15681763415, y=0.91973859035642)
        y = Complex(x=-314598817495.9188910355, y=34256.413254268910254)
        assert x * y == Complex(x=-6.011525081204529213e18, y=3.652414781354840040e11)

    def test_five(self):
        x = Complex(x=6, y=8)
        y = Complex(x=-6, y=-8)
        assert x * y == Complex(x=28, y=-96)

    def test_six(self):
        x = Complex(
            x=922548813.4133333389547892, y=9147385763256475.1245354981743561754
        )
        y = Complex(x=0, y=0)
        assert x * y == Complex(x=0, y=0)


class TestDiv:
    def test_one(self):
        x = Complex(x=1, y=4)
        y = Complex(x=3, y=-8)
        assert x / y == Complex(x=-29 / 73, y=20 / 73)

    def test_two(self):
        x = Complex(x=98457635648, y=-2523449274358385687432)
        y = Complex(x=67813493425, y=76273598073625)
        assert x / y == Complex(x=-3.3084151572613217e7, y=-29414.5294081065901)

    def test_three(self):
        x = Complex(x=0, y=1)
        y = Complex(x=1, y=-0)
        assert x / y == Complex(x=0, y=1)

    def test_four(self):
        x = Complex(x=19108543.15681763415, y=0.91973859035642)
        y = Complex(x=-314598817495.9188910355, y=34256.413254268910254)
        assert x / y == Complex(x=-0.00006073939917801881611, y=-9.537392967008077e-12)

    def test_five(self):
        x = Complex(x=6, y=8)
        y = Complex(x=-6, y=-8)
        assert x / y == Complex(x=-1, y=0)

    def test_six(self):
        x = Complex(
            x=922548813.4133333389547892, y=9147385763256475.1245354981743561754
        )
        y = Complex(x=0, y=0)
        with pytest.raises(ZeroDivisionError) as error_info:
            x / y
        assert error_info.match(r"The denominator is zero")


class TestPow:
    def test_one(self):
        x = Complex(x=1, y=4)
        y = Complex(x=3, y=-8)
        assert x**y == Complex(
            x=1.35402642167989307613443827e6,
            y=-2.4869082896360332436468119255517013395977e6,
        )

    @pytest.mark.xfail
    def test_two(self):
        x = Complex(x=98435425763, y=-2523345324544927)
        y = Complex(x=6781342534, y=7627345345359)
        assert x**y == Complex(x=0, y=0)

    def test_three(self):
        x = Complex(x=0, y=1)
        y = Complex(x=1, y=-0)
        assert x**y == Complex(x=0, y=1)

    def test_four(self):
        x = Complex(x=193.15681763, y=0.91973642)
        y = Complex(x=-31745.9188915, y=34256.41268910254)
        assert x**y == Complex(x=4.82313e-72460, y=8.05908e-72641)

    def test_five(self):
        x = Complex(x=6, y=8)
        y = Complex(x=-6, y=-8)
        assert x**y == Complex(x=0.0006832642737281139, y=0.0015197731502737256349)

    def test_six(self):
        x = Complex(
            x=922548813.4133333389547892, y=9147385763256475.1245354981743561754
        )
        y = Complex(x=0, y=0)
        assert x**y == Complex(x=1, y=0)
