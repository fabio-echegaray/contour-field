import logging
import time

import numpy as np
from shapely.geometry import Polygon
from skimage.transform import PiecewiseAffineTransform, warp
from symfit import Eq, Fit, cos, parameters, pi, sin, variables
from scipy.interpolate import UnivariateSpline

from gui._image_loading import retrieve_image
from gui.measure import FileImageMixin

logger = logging.getLogger('rectification')


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class BaseApproximation(FileImageMixin):
    def __init__(self, polygon: Polygon, image):
        super(FileImageMixin, self).__init__()
        self._load_image(image)
        self._poly = polygon

        self._fn = lambda x: np.array([0, 0])
        self._dfn_dt = lambda x: np.array([0, 0])

    def _load_image(self, img):
        if type(img) is str:
            self.file = img
        elif issubclass(type(img), FileImageMixin):
            self.fileimage_from(img)
        else:
            raise Exception("Couldn't load image file.")

    def approximate_fn(self):
        pass

    def f(self, t):
        return self._fn(t)

    def tangent_angle(self, t):
        dx, dy = self._dfn_dt(t)
        if dy != 0:
            return np.arctan2(dy, dx)
        else:
            return np.nan

    def normal_angle(self, t):
        dx, dy = self._dfn_dt(t)
        if dx != 0:
            # as arctan2 argument order is  y, x (and as we're doing a rotation) -> x=-dy y=dx)
            return np.arctan2(dx, -dy)
        else:
            return np.nan


@timeit
def harmonic_approximation(polygon: Polygon, n=3):
    def fourier_series(x, f, n=0):
        """
        Returns a symbolic fourier series of order `n`.

        :param n: Order of the fourier series.
        :param x: Independent variable
        :param f: Frequency of the fourier series
        """
        # Make the parameter objects for all the terms
        a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
        sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
        # Construct the series
        series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                          for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
        return series

    x, y = variables('x, y')
    w, = parameters('w')
    fourier = fourier_series(x, f=w, n=n)
    model_dict = {y: fourier}
    print(model_dict)

    # Extract data from argument
    # FIXME: how to make a clockwise strictly increasing curve?
    xdata, ydata = polygon.exterior.xy
    t = np.linspace(0, 2 * np.pi, num=len(xdata))

    constr = [
        # Ge(x, 0), Le(x, 2 * pi),
        Eq(fourier.subs({x: 0}), fourier.subs({x: 2 * pi})),
        Eq(fourier.diff(x).subs({x: 0}), fourier.diff(x).subs({x: 2 * pi})),
        # Eq(fourier.diff(x, 2).subs({x: 0}), fourier.diff(x, 2).subs({x: 2 * pi})),
    ]
    print(constr)

    fit_x = Fit(model_dict, x=t, y=xdata, constraints=constr)
    fit_y = Fit(model_dict, x=t, y=ydata, constraints=constr)
    fitx_result = fit_x.execute()
    fity_result = fit_y.execute()
    print(fitx_result)
    print(fity_result)

    # Define function that generates the curve
    def curve_lambda(_t):
        return np.array(
            [
                fit_x.model(x=_t, **fitx_result.params).y,
                fit_y.model(x=_t, **fity_result.params).y
            ]
        ).ravel()

    # code to test if fit is correct
    plot_fit(polygon, curve_lambda, t, title='Harmonic Approximation')

    return curve_lambda


class SplineApproximation(BaseApproximation):
    def __init__(self, polygon: Polygon, image):
        super(SplineApproximation, self).__init__(polygon, image)
        self.approximate_fn()

    @timeit
    def approximate_fn(self):
        # Extract data from argument
        # FIXME: how to make a clockwise strictly increasing curve?
        cols, rows = self._poly.exterior.xy
        t = np.linspace(0, 2 * np.pi, num=len(cols))

        # Define spline minimizer function
        splx = UnivariateSpline(t, cols)
        sply = UnivariateSpline(t, rows)
        # splx.set_smoothing_factor(2 * self.pix_per_um)
        # sply.set_smoothing_factor(2 * self.pix_per_um)

        # Define spline 1st order derivative
        dsplx_dt = splx.derivative()
        dsply_dt = sply.derivative()

        # Define function that generates the curve
        self._fn = lambda o: np.array([splx(o), sply(o)])
        self._dfn_dt = lambda o: np.array([dsplx_dt(o), dsply_dt(o)])


class TestSplineApproximation(SplineApproximation):
    def test_fit(self):
        t = np.linspace(0, 2 * np.pi, num=len(self._poly.exterior.xy[0]))
        plot_fit(self._poly, self.f, t, title='Spline Approximation')

    def plot_rectification(self):
        import matplotlib.pyplot as plt
        image = retrieve_image(self.images, channel=0, number_of_channels=self.nChannels,
                               zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)

        rows, cols = image.shape[0], image.shape[1]

        # rectify the image using a piecewise affine transform from skimage libray
        # define the ending points of the transformation
        dl_dom = np.linspace(-1, 1, num=10) * self.pix_per_um
        t_dom = np.linspace(0, 2 * np.pi, num=200)
        dst_rows, dst_cols = np.meshgrid(dl_dom, t_dom)

        # calculate the original points
        src_rows = dst_rows.copy()
        src_cols = dst_cols.copy()
        for i in range(src_cols.shape[0]):
            print(src_cols[i, :])
            t = src_cols[i, 0]
            x0, y0 = self.f(t)
            o = self.normal_angle(t)
            sin_o = np.sin(o)
            cos_o = np.cos(o)

            for j in range(src_rows.shape[1]):
                dl = src_rows[i, j]
                print(f"debug i={j},  j={i}, dl={dl:.2f}   "
                      f"src_cols[i, j]-t={src_cols[i, j] - t:.3f}")
                src_cols[i, j] = x0 + dl * cos_o
                src_rows[i, j] = y0 + dl * sin_o

        # rescale the point of the dst mesh to match output image
        out_rows = dl_dom.size * 100
        out_cols = t_dom.size * 100
        dst_rows = np.linspace(0, out_rows, dl_dom.size)
        dst_cols = np.linspace(0, out_cols, t_dom.size)
        dst_rows, dst_cols = np.meshgrid(dst_rows, dst_cols)

        # convert meshes tho (N,2) vectors
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        dst = np.dstack([dst_cols.flat, dst_rows.flat])[0]

        tform = PiecewiseAffineTransform()
        tform.estimate(dst, src)

        out = warp(image, tform, output_shape=(out_rows, out_cols))  # , order=2)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image, origin='lower')
        ax1.plot(src[:, 0], src[:, 1], '.b')
        ax1.axis((0, cols, rows, 0))

        ax2.imshow(out, origin='lower')
        ax2.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
        ax2.axis((0, out_cols, out_rows, 0))

        fig = plt.figure(20)
        ax = fig.gca()
        ext = [0, t_dom.max(), dl_dom.max() * 2, 0]
        plt.imshow(out, origin='lower', extent=ext, aspect='auto')
        ax.set_title('Image rectification')

        plt.show()

    def plot_grid(self):
        import matplotlib.pyplot as plt
        import plots as p

        fig = plt.figure(10)
        ax = fig.gca()

        c = self._poly.centroid
        xdata, ydata = self._poly.exterior.xy
        ax.set_xlim([min(xdata) - 2 * self.pix_per_um, max(xdata) + 2 * self.pix_per_um])
        ax.set_ylim([min(ydata) - 2 * self.pix_per_um, max(ydata) + 2 * self.pix_per_um])

        dna_ch = 0
        act_ch = 2
        image = retrieve_image(self.images, channel=dna_ch, number_of_channels=self.nChannels,
                               zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
        ax.imshow(image, origin='lower', cmap='gray')

        p.render_polygon(self._poly, zorder=10, ax=ax)
        pts = np.array([self.f(_t) for _t in np.linspace(0, 2 * np.pi, num=len(xdata))])
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c='blue')
        ax.set_title("Grid on image")

        theta = np.linspace(0, 2 * np.pi, num=20)
        fx, fy = self.f(theta)
        ax.scatter(fx, fy, linewidth=1, marker='o', edgecolors='yellow', facecolors='none', label='x')

        # plot normals and tangents
        dl_arr = np.linspace(-1, 1, num=5) * self.pix_per_um
        # dl_arr = np.linspace(0, 1, num=3) * self.pix_per_um
        for i, t in enumerate(theta):
            x0, y0 = self.f(t)

            o = self.normal_angle(t)
            fnx, fny = np.array([(x0 + dl * np.cos(o), y0 + dl * np.sin(o)) for dl in dl_arr]).T
            ax.plot(fnx, fny, linewidth=1, linestyle='-', c='white', marker='o', ms=1)

            th = self.tangent_angle(t)
            ftx, fty = np.array([(x0 + dl * np.cos(th), y0 + dl * np.sin(th)) for dl in dl_arr]).T
            ax.plot(ftx, fty, linewidth=1, linestyle='-', c='red')

            plt.annotate((f"{i} ({t:.2f}): "
                          f"T{np.rad2deg(th):.0f}º  "
                          f"N{np.rad2deg(o):.0f}º "
                          f"{np.sin(o):.2f} {np.cos(o):.2f}"), (x0, y0),
                         (min(xdata) if x0 < c.x else max(xdata), y0), color="red",
                         horizontalalignment='right' if x0 < c.x else 'left',
                         arrowprops=dict(facecolor='white', shrink=0.05))

        plt.show()


def plot_fit(polygon: Polygon, fit_fn, t, title=""):
    import matplotlib.pyplot as plt
    import plots as p

    xdata, ydata = polygon.exterior.xy
    pts = np.array([fit_fn(_t) for _t in t])

    fig = plt.figure(10)
    ax = fig.gca()

    ax.scatter(t, xdata, linewidth=1, marker='o', edgecolors='red', facecolors='none', label='x')
    ax.scatter(t, ydata, linewidth=1, marker='o', edgecolors='blue', facecolors='none', label='y')
    ax.plot(t, pts[:, 0], linewidth=1, linestyle='-', c='red')
    ax.plot(t, pts[:, 1], linewidth=1, linestyle='-', c='blue')
    ax.set_title(title)
    ax.legend()

    fig = plt.figure(20)
    ax = fig.gca()
    # plt.imshow(image, origin='lower')
    # n_um = affinity.scale(n, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

    p.render_polygon(polygon, zorder=10, ax=ax)
    ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c='blue')
    ax.set_title('Spline approximation')

    plt.show()
