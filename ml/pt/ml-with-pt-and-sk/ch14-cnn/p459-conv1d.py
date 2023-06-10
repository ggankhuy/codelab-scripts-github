
import code
import numpy as np

def conv1d(x, w, p=0, s=1):

    # w_rot: reverses the w.

    w_rot = np.array(w[::-1])

    # prepares padded version of x, not padded yet.

    x_padded = np.array(x)

    if p > 0:

        # prepares pad amount, same as p.
       
        zero_pad = np.zeros(shape=p)

        # add padding (border) to both side of x.

        x_padded = np.concatenate([ zero_pad, x_padded, zero_pad ])
        res=[]
        
        # for each i which i from 0 to len of x_padded - w + 1, stride is step amount.
        # multiply each member of x_padded from i to i+len(w) by w_rot. this is element by element multiplication.
        # once multiple sum it to scalar.
        # add to res.

        for i in range(0, int((len(x_padded) - len(w_rot))) + 1, s):
            res.append(np.sum(x_padded[i:i+w_rot.shape[0]] * w_rot))
        return np.array(res)

x=[1,3,2,4,5,6,1,3]
w=[1,0,3,1,2]

result=conv1d(x,w, p=2, s=1)
print("Conv1d implementation: ", conv1d(x,w, p=2, s=1))
print('NumPy Results: ', np.convolve(x,w,mode='same'))

#code.interact(local=locals())
