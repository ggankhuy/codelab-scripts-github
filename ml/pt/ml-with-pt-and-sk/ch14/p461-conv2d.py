# seeing decrecated warning and screen filled with endless garbage when executing...

import numpy as np
import code

def conv2d(x, w, p=(0,0), s=(1,1)):
    w_rot=np.array(w)[::-1,::-1]
    x_orig = np.array(x)
    n1=x_orig.shape[0] + 2 * p[0]
    n2=x_orig.shape[1] + 2 * p[1]

    x_padded = np.zeros(shape=(n1, n2))
    x_padded[p[0]:p[0] + x_orig.shape[0], \
             p[1]:p[1] + x_orig.shape[1]] = x_orig

    res=[]

    for i in range(0, int((x_padded.shape[0] - w_rot.shape[0]) / s[0]) + 1, s[0]):
        res.append([])
        for j in range(0, int((x_padded.shape[1] - w_rot.shape[1]) / s[1]) + 1, s[1]):
            x_sub = x_padded[i:i+w_rot.shape[0], j:j+w_rot.shape[1]]
            res.append(np.array(res))

    return(np.array(res))

x=[[1,3,2,4], [5,6,1,3],[1,2,0,2], [3,4,3,2]]
w=[[1,0,3], [1,2,1], [0,1,1]]
print('Conv2d implementation: \n', conv2d(x,w, (1,1), s=(1,1)))        
print("sciPy results: \n", scipy.signal.convolve2d(x,w,mode='same'))
#code.interact(local=locals())


