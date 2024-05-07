a=123
import torch
x=torch.as_tensor([[1,2],[3,4]])
import sys
sys.path.append('../..')

from common.settings import *
from common.classes import *

def namestrLocal(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def printTensorLocal(pVarName):
    print('--------------------------------')
    print("globals: ")
    #for i in globals():
    #    print(i)
    print(namestr(pVarName, globals()))
    print('--------------------------------')

printTensorLocal(x)
printTensor(x)
printTensor(x, globals())


# printTensorLocal(x)->nameStr() ok
# printTensorLocal(x)->nameStrLocal() ok
# printTensor(x)->nameStr() fail

