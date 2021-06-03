import numpy as np
from scipy.interpolate import interp1d  # scipyのモジュールを使う
import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.5):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def make_func(in_x):
    np.random.seed(0)
    out_y = np.exp(-in_x) + 0.4*(np.random.rand(in_x.size) -
                                 0.5)  # exp(-x)の式にランダムな誤差を入れる
    return out_y


def spline_interp(in_x, in_y):
    out_x = np.linspace(np.min(in_x), np.max(
        in_x), np.size(in_x)*2)  # もとのxの個数より多いxを用意
    func_spline = interp1d(in_x, in_y, kind='cubic')  # cubicは3次のスプライン曲線
    out_y = func_spline(out_x)  # func_splineはscipyオリジナルの型

    return out_x, out_y
