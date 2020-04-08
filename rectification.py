import logging
import time

import numpy as np
from shapely.geometry import Polygon
from skimage.transform import PiecewiseAffineTransform, warp
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


# FIXME: how to make a clockwise strictly increasing curve? This is:
#  1 - how to homogenize the points so it won't have more dense areas?
#  2 - how to make the parametrization function of the arclength?

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
        if dx.any() != 0:
            # as arctan2 argument order is  y, x (and as we're doing a rotation) -> x=-dy y=dx)
            return np.arctan2(dx, -dy)
        else:
            return np.nan


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


class FunctionRectification:
    # rectify the image using the approximated function directly
    def __init__(self, curve: BaseApproximation, dl=1, n_dl=10, n_theta=50, pix_per_dl=100, pix_per_theta=100):
        self._model = curve

        self.dl = dl
        self.n_dl = n_dl
        self.n_theta = n_theta
        self.pix_per_dl = pix_per_dl
        self.pix_per_theta = pix_per_theta

        self.out_rows = n_dl * pix_per_dl
        self.out_cols = n_theta * pix_per_theta

    def curve(self, cr_i):
        theta = cr_i[0] / self.pix_per_theta / self.n_theta * 2 * np.pi
        dl = cr_i[1] / self.pix_per_dl / self.n_dl * 2 * self.dl - self.dl
        x0, y0 = self._model.f(theta)
        o = self._model.normal_angle(theta)
        _x = x0 + dl * np.cos(o)
        _y = y0 + dl * np.sin(o)
        return _x, _y

    @timeit
    def rectify(self, image):
        def rect_fn(cr: np.array):
            # a function that transforms a (M, 2) array of (col, row)
            return np.apply_along_axis(self.curve, axis=1, arr=cr)

        return warp(image, rect_fn, output_shape=(self.out_rows, self.out_cols))  # , order=2)

    # def curve(self, cr: np.array):
    #     # a function that transforms a (M, 2) array of (col, row)
    #     theta = cr[:, 0] / self.pix_per_theta / self.n_theta * 2 * np.pi
    #     dl = cr[:, 1] / self.pix_per_dl / self.n_dl * 2 * self.dl - self.dl
    #     x0, y0 = self._model.f(theta)
    #     o = self._model.normal_angle(theta)
    #     cols = x0 + dl * np.cos(o)
    #     rows = y0 + dl * np.sin(o)
    #     return np.array([cols, rows])
    #
    # @timeit
    # def rectify(self, image):
    #     return warp(image, self.curve, output_shape=(self.out_rows, self.out_cols))  # , order=2)


class TestFunctionRectification(FunctionRectification):
    def plot_rectification(self):
        import matplotlib.pyplot as plt
        image = retrieve_image(self._model.images, channel=0, number_of_channels=self._model.nChannels,
                               zstack=self._model.zstack, number_of_zstacks=self._model.nZstack, frame=0)

        rows, cols = image.shape[0], image.shape[1]
        out = self.rectify(image)

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
        ax1.imshow(image, origin='lower')
        # ax1.plot(self.src[:, 0], self.src[:, 1], '.b')
        ax1.axis((0, cols, rows, 0))

        ax2.imshow(out, origin='lower')
        # ax2.plot(self.transform.inverse(self.src)[:, 0], self.transform.inverse(self.src)[:, 1], '.b')
        ax2.axis((0, self.out_cols, self.out_rows, 0))

        fig = plt.figure()
        ax = fig.gca()
        # ext = [0, t_dom.max(), dl_dom.max() * 2, 0]
        plt.imshow(out, origin='lower', aspect='auto')  # , extent=ext)
        ax.set_title('Image rectification using original function')

        plt.show(block=False)


class PiecewiseLinearRectification:
    # rectify the image using a piecewise affine transform from skimage library
    def __init__(self, curve: BaseApproximation, dl=1, n_dl=10, n_theta=50, pix_per_dl=100, pix_per_theta=100):
        self._model = curve
        self.dl_dom = None
        self.t_dom = None
        self.src = None
        self.dst = None
        self.out_rows = None
        self.out_cols = None
        self.dst_rows = None
        self.dst_cols = None

        self.estimated = False

        self.dl = dl
        self.n_dl = n_dl
        self.n_theta = n_theta
        self.pix_per_dl = pix_per_dl
        self.pix_per_theta = pix_per_theta

    @timeit
    def _estimate_transform(self):
        if self.estimated:
            return

        self.estimated = True

        # define the ending points of the transformation
        self.dl_dom = np.linspace(-self.dl, self.dl, num=self.n_dl) * self._model.pix_per_um
        self.t_dom = np.linspace(0, 2 * np.pi, num=self.n_theta)
        self.dst_rows, self.dst_cols = np.meshgrid(self.dl_dom, self.t_dom)

        # calculate the original points
        self.src_rows = self.dst_rows.copy()
        self.src_cols = self.dst_cols.copy()
        for i in range(self.src_cols.shape[0]):
            t = self.src_cols[i, 0]
            x0, y0 = self._model.f(t)
            o = self._model.normal_angle(t)
            sin_o = np.sin(o)
            cos_o = np.cos(o)

            for j in range(self.src_rows.shape[1]):
                dl = self.src_rows[i, j]
                logger.debug(f"debug i={j},  j={i}, dl={dl:.2f}   src_cols[i, j]-t={self.src_cols[i, j] - t:.3f}")
                self.src_cols[i, j] = x0 + dl * cos_o
                self.src_rows[i, j] = y0 + dl * sin_o

        # rescale the point of the dst mesh to match output image
        self.out_rows = self.dl_dom.size * self.pix_per_dl
        self.out_cols = self.t_dom.size * self.pix_per_theta
        self.dst_rows = np.linspace(0, self.out_rows, self.dl_dom.size)
        self.dst_cols = np.linspace(0, self.out_cols, self.t_dom.size)
        self.dst_rows, self.dst_cols = np.meshgrid(self.dst_rows, self.dst_cols)

        # convert meshes to (N,2) vectors
        self.src = np.dstack([self.src_cols.flat, self.src_rows.flat])[0]
        self.dst = np.dstack([self.dst_cols.flat, self.dst_rows.flat])[0]

        self.transform = PiecewiseAffineTransform()
        self.transform.estimate(self.dst, self.src)

    @timeit
    def rectify(self, image):
        self._estimate_transform()
        return warp(image, self.transform, output_shape=(self.out_rows, self.out_cols))  # , order=2)


class TestPiecewiseLinearRectification(PiecewiseLinearRectification):
    def plot_rectification(self):
        import matplotlib.pyplot as plt
        image = retrieve_image(self._model.images, channel=0, number_of_channels=self._model.nChannels,
                               zstack=self._model.zstack, number_of_zstacks=self._model.nZstack, frame=0)

        rows, cols = image.shape[0], image.shape[1]
        out = self.rectify(image)

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
        ax1.imshow(image, origin='lower')
        # ax1.plot(self.src[:, 0], self.src[:, 1], '.b')
        ax1.axis((0, cols, rows, 0))

        ax2.imshow(out, origin='lower')
        # ax2.plot(self.transform.inverse(self.src)[:, 0], self.transform.inverse(self.src)[:, 1], '.b')
        ax2.axis((0, self.out_cols, self.out_rows, 0))

        fig = plt.figure()
        ax = fig.gca()
        # ext = [0, t_dom.max(), dl_dom.max() * 2, 0]
        plt.imshow(out, origin='lower', aspect='auto')  # , extent=ext)
        ax.set_title('Image rectification using piecewise linear transform')

        plt.show(block=False)


class TestSplineApproximation(SplineApproximation):
    def test_fit(self):
        t = np.linspace(0, 2 * np.pi, num=len(self._poly.exterior.xy[0]))
        plot_fit(self._poly, self.f, t, title='Spline Approximation')

    def plot_grid(self):
        import matplotlib.pyplot as plt
        import plots as p

        fig = plt.figure()
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

        plt.show(block=False)


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

    fig = plt.figure()
    ax = fig.gca()
    # plt.imshow(image, origin='lower')
    # n_um = affinity.scale(n, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

    p.render_polygon(polygon, zorder=10, ax=ax)
    ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c='blue')
    ax.set_title('Spline approximation')

    plt.show(block=False)
