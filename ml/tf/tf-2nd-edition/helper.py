import re
import time
import numpy as np

def process_params(pSysArg, pList):
    DEBUG=1
    ret=[]
    for i in pSysArg:
        if DEBUG:
            print("Processing ", i)
        try:
            for j in pList:
                if re.search(j, i + "="):
                    try:
                        ret.append(int(i.split('=')[1]))
                    except Exception as msg:
                        ret.append(i.split('=')[1])
                
        except Exception as msg:
            print("Error processing ", i)
            print(msg)

    if DEBUG:   
        print("returning ", ret)

    return ret

        
def generate_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    print("freq1, freq2, offset1, offset2: ", freq1.shape, freq2.shape, offsets1.shape, offsets2.shape)
    time1 = np.linspace(0, 1, n_steps)
    print("time1: ", time1.shape)

    # wave 1

    series = 0.5 * np.sin((time1 - offsets1) * (freq1 * 10 + 10))

    # wave 2

    series += 0.3 * np.sin((time1 - offsets2) * (freq2 * 20 + 20))

    # noise

    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    print("returning series (shape): ", series.shape)
    return series[..., np.newaxis].astype(np.float32)

