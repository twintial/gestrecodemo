import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


# unwrap phase
def unwrap(data):
    return np.unwrap(data)



# 找最大的k个峰值的索引
def find_kmax_peaks(data, k):
    peak_index, _ = find_peaks(data, distance=5)
    position = max_k_index(data[peak_index], k)
    return np.array(peak_index[position])

# 插值
def interp(t, s, range, T, kind="linear"):
    f_linear = interp1d(t, s, kind, fill_value='extrapolate')
    tt = np.linspace(range[0], range[1], (range[1] - range[0])/T)
    ss = f_linear(tt)
    return tt, ss

def resampling(t, s, range, num, kind="linear"):
    f_linear = interp1d(t, s, kind, fill_value='extrapolate')
    tt = np.linspace(range[0], range[1], int(num))
    ss = f_linear(tt)
    return tt, ss

# 找最大的k个值
def max_k(data, k):
    sorted_data = sorted(data, reverse=True)
    return sorted_data[0: k]


# 找最大的k个值的索引
def max_k_index(data, k):
    max1, max2 = max_k(data, k)
    return np.array([data.tolist().index(max1), data.tolist().index(max2)])


def interp1(t, s, T, kind='linear'):
    tt, ss = interp(t, s, [t[0], t[t.size - 1]], T, kind)
    # f_linear = interp1d(t, s, kind)
    # min_t = t[0]
    # max_t = t[t.size - 1]
    # tt = np.linspace(min_t, max_t, (max_t - min_t)/T)
    # ss = f_linear(tt)

    # plt.figure()
    # plt.plot(t, s, color='blue', label='original')
    # plt.plot(tt, ss, color='red', label='after interp')
    # plt.show()
    return tt, ss


# 0:max-min 1:std
def moving_std(x, wlen, hop, method):
    result = []
    xlen = len(x)
    coln = 1 + (xlen-wlen) // hop
    indx = 0
    for i in range(0, coln-1, 1):
        if method == 1:
            value = np.std(x[indx:(indx + wlen)])
        else:
            value = max(x[indx:(indx + wlen)]) - min(x[indx:(indx + wlen)])
        result.append(value)
        indx = indx + hop
    index = np.array(range(wlen//2, wlen//2+(coln-1)*hop, hop))
    return index, result


# filter DC
def moving_average(s, k=3):
    n = s.size
    result = []
    for i in range(n):
        if i < k:
            window = [0, min(2*i, n - 1)]
        elif i > n - k - 1:
            window = [i, n - 1]
        else:
            window = [i-k, i+k]
        result.append(np.mean(s[window]))
    return np.array(result)


# moving average
# def moving_average(a, n=3):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

# hampel

if __name__ == "__main__":
    pass
    # test moving_std
    # x = range(100)
    # y = np.sin(x)
    # t, result = moving_std(y, 7, 1, 1)
    # plt.figure()
    # plt.plot(t, result, color='blue', label='moving_std')
    # plt.show()
