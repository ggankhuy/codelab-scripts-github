a=123
import torch
x=torch.as_tensor([[1,2],[3,4]])
import sys
sys.path.append('../..')

from common.settings import *
from common.classes import *
class tmp1:
    tmp1_x = 10
    tmp1_y = [0,1,2,3]

    def tmp1_fn1(self):

        print("printing namespaces from within a class...")
        #print(self.__dict__)

        printTensor(self.tmp1_x, getGlobalsClass(self))
        printTensor(self.tmp1_y, getGlobalsClass(self))

def getGlobalsClass(pObj):
    print("getGlobalClass entered...")
    d1={}
    for i in dir(pObj):
        d1[i] = getattr(pObj,i)
    print("returning: ", type(d1))
    for i in d1.keys():
        print (" -- ", i, d1[i])

    return d1
    
def namestrLocal(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def printTensorLocal(pVarName):
    print('--------------------------------')
    print("globals: ")
    #for i in globals():
    #    print(i)
    print(namestr(pVarName, globals()))
    print('--------------------------------')

'''
printTensorLocal(x)
printTensor(x)
printTensor(x, globals())
'''
tmp1_inst = tmp1()
tmp1_inst.tmp1_fn1()

# printTensorLocal(x)->nameStr() ok
# printTensorLocal(x)->nameStrLocal() ok
# printTensor(x)->nameStr() fail

