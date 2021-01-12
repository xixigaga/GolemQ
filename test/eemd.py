# python 实现对沪深300的 EEMD分解 解析出不同级别的自回归周期频率
import numpy as np
from PyEMD import EEMD, EMD, Visualisation
import pylab as plt

from GolemQ.fetch.kline import (
    get_kline_price,
    get_kline_price_min,
)
from GolemQ.fractal.v0 import (
    maxfactor_cross_v2_func,
)
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
)

if __name__ == "__main__":
    kline, display_name = get_kline_price_min('399300', frequency='60min',)
    features = maxfactor_cross_v2_func(kline.data)

    max_imf = 17
    S = features['CCI'].dropna().tail(480).values
    T = range(len(S))

    # EEMD计算
    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(12345)

    E_IMFs = eemd.eemd(S, T, max_imf)
    imfNo = E_IMFs.shape[0]
    tMin, tMax = np.min(T), np.max(T)

    # Plot results in a grid
    c = np.floor(np.sqrt(imfNo + 1))
    r = np.ceil((imfNo + 1) / c)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(T, S, 'r')
    plt.xlim((tMin, tMax))
    plt.title("399300.XSHE:CCI(14)")

    for num in range(imfNo):
        plt.subplot(r, c, num + 2)
        plt.plot(T, E_IMFs[num], 'g')
        plt.xlim((tMin, tMax))
        plt.title("Imf " + str(num + 1))

    plt.show()