# seeing decrecated warning and screen filled with endless garbage when executing...

import numpy as np
import code

def conv2d(x, w, p=(0,0), s=(1,1)):

    # reverse each dimension of w, inner and outer dimension.

    w_rot=np.array(w)[::-1,::-1]

    x_orig = np.array(x)

    # each dimension of x a two by default. Note that p=(1,1) by external call. 4+2=6 in this instance.
    # multiply by 2 is due to adding both side.

    n1=x_orig.shape[0] + 2 * p[0]
    n2=x_orig.shape[1] + 2 * p[1]

    # prepare x padded version. [6,6]. x_orig (4,4) will be at the center of 6,6 in the range(1:4, 1:4)

    x_padded = np.zeros(shape=(n1, n2))
    x_padded[p[0]:p[0] + x_orig.shape[0], \
             p[1]:p[1] + x_orig.shape[1]] = x_orig

    res=[]

    # walk along 1st dimension from 0 to len(x_padded=6 - w=3) increment by stride amount (1). 

    for i in range(0, int((x_padded.shape[0] - w_rot.shape[0]) / s[0]) + 1, s[0]):

        # add another empty 2nd dimension. 

        res.append([])

        # walk along 2nd dimension from 0 to len(x_padded=6 - w=3) increment by stride amount (1). 

        for j in range(0, int((x_padded.shape[1] - w_rot.shape[1]) / s[1]) + 1, s[1]):

            # x_sub: window from i to i+len(w) along both dimensions.

            x_sub = x_padded[i:i+w_rot.shape[0], j:j+w_rot.shape[1]]

            # append to res's inner dimension.

            res[-1].append(np.sum(x_sub *  w_rot))

    return(np.array(res))

x=[[1,3,2,4], [5,6,1,3],[1,2,0,2], [3,4,3,2]]
w=[[1,0,3], [1,2,1], [0,1,1]]
result=conv2d(x,w, (1,1), s=(1,1))

print('Conv2d implementation: \n', conv2d(x,w, (1,1), s=(1,1)))        
print("sciPy results: \n", scipy.signal.convolve2d(x,w,mode='same'))
#code.interact(local=locals())


