# Copyright ©2022 Zhang Xinyu, Sun weiyu and Chen ying. All Rights Reserved.
import numpy as np
from utils.cwtbag import cwt_filtering
from scipy import signal

def peakcheckez(a, samplingrate):
    result = []
    for i in range(len(a)):
        if i == 0 or i == len(a) - 1:
            pass
        else:
            if a[i] >= a[i - 1] and a[i] > a[i + 1]:
                result.append(i)

    hr_list = []
    if len(result) <= 1:
        hr = 0
    else:
        for i in range(len(result) - 1):
            hr = 60 * samplingrate / (result[i + 1] - result[i])
            hr_list.append(hr)
        hr = np.mean(np.array(hr_list))
    return hr


def hr_cal(tmp, samplingrate=30):
    f1 = 0.5
    f2 = 3
    samplingrate = samplingrate
    b, a = signal.butter(4, [2 * f1 / samplingrate, 2 * f2 / samplingrate], 'bandpass')
    tmp = signal.filtfilt(b, a, np.array(tmp))
    tmp = cwt_filtering(tmp, samplingrate)[0]

    hr_caled = peakcheckez(tmp, samplingrate)
    return hr_caled, tmp
