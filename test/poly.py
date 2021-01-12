def strided_app(a, L, S):  
    '''
    Pandas rolling for numpy
    # Window len = L, Stride len/stepsize = S
    '''
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S * n,n))

def rolling_poly9(s:np.ndarray, w:int=252) -> np.ndarray:
    '''
    一次九项式滚动分解拟合
    '''
    x_index = range(252)
    def last_poly9(sx):
        p = np.polynomial.Chebyshev.fit(x_index, sx, 9)
        return p(x_index)[-1]

    if (len(s) > w):
        x = strided_app(s, w, 1)
        return np.r_[np.full(w - 1, np.nan), 
                     np.array(list(map(last_poly9, x)))]
    else:
        x_index = range(len(s))
        p = np.polynomial.Chebyshev.fit(x_index, s, 9)
        y_fit_n = p(x_index)
        return y_fit_n

# 用法示例
features['POLYNOMIAL9'] = rolling_poly9(features['HMA10'].values, 252)