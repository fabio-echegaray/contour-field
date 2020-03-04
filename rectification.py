import logging
import time

import numpy as np
from shapely.geometry import Polygon
from symfit import Eq, Fit, cos, parameters, pi, sin, variables
from scipy.interpolate import UnivariateSpline

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

    # # code to test if fit is correct
    # import matplotlib.pyplot as plt
    # import plots as p
    #
    # fig = plt.figure(10)
    # ax = fig.gca()
    # # plt.imshow(image, origin='lower')
    # # n_um = affinity.scale(n, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))
    #
    # pts = np.array([curve_lambda(x) for x in t])
    # # print(pts)
    # # print(pts.shape)
    #
    # ax.scatter(t, xdata, linewidth=1, marker='+', edgecolors='red', facecolors='none', label='x')
    # ax.scatter(t, ydata, linewidth=1, marker='+', edgecolors='blue', facecolors='none', label='y')
    # ax.plot(t, pts[:, 0], linewidth=1, linestyle='-.', c='red')
    # ax.plot(t, pts[:, 1], linewidth=1, linestyle='-.', c='blue')
    # ax.set_title('Harmonic approximation')
    # ax.legend()
    #
    # fig = plt.figure(20)
    # ax = fig.gca()
    # # plt.imshow(image, origin='lower')
    # # n_um = affinity.scale(n, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))
    #
    # p.render_polygon(polygon, zorder=10, ax=ax)
    # ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c='blue')
    # ax.set_title('Harmonic approximation')
    #
    # plt.show()

    return curve_lambda


@timeit
def spline_approximation(polygon: Polygon, n=3, pix_per_um=1):
    # Extract data from argument
    # FIXME: how to make a clockwise strictly increasing curve?
    xdata, ydata = polygon.exterior.xy
    t = np.linspace(0, 2 * np.pi, num=len(xdata))

    # Define spline minimizer function
    splx = UnivariateSpline(t, xdata)
    sply = UnivariateSpline(t, ydata)

    # splx.set_smoothing_factor(0.5)
    # sply.set_smoothing_factor(0.5)
    # print(splx.get_coeffs(), splx.get_coeffs().shape)
    # print(sply.get_knots(), splx.get_knots().shape)

    # Define function that generates the curve
    def curve_lambda(_t):
        return np.array([splx(_t), sply(_t)])

    # pts = np.array([curve_lambda(x) for x in t])
    #
    # # convert to polar coordinates
    # px, py = polygon.centroid.xy
    # r = np.sqrt((pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2)
    # r -= 2 * pix_per_um
    # pts = np.array([r * np.cos(t) + px, r * np.sin(t) + py]).T
    # print(pts.shape)

    # # code to test if fit is correct
    # import matplotlib.pyplot as plt
    # import plots as p
    #
    # fig = plt.figure(10)
    # ax = fig.gca()
    #
    # # print(pts)
    # # print(pts.shape)
    #
    # ax.scatter(t, xdata, linewidth=1, marker='o', edgecolors='red', facecolors='none', label='x')
    # ax.scatter(t, ydata, linewidth=1, marker='o', edgecolors='blue', facecolors='none', label='y')
    # ax.plot(t, pts[:, 0], linewidth=1, linestyle='-', c='red')
    # ax.plot(t, pts[:, 1], linewidth=1, linestyle='-', c='blue')
    # ax.set_title('Spline approximation')
    # ax.legend()
    #
    # fig = plt.figure(20)
    # ax = fig.gca()
    # # plt.imshow(image, origin='lower')
    # # n_um = affinity.scale(n, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))
    #
    # p.render_polygon(polygon, zorder=10, ax=ax)
    # ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c='blue')
    # ax.set_title('Spline approximation')
    #
    # plt.show()

    return curve_lambda
