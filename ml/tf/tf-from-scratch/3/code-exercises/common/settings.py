import inspect
import numpy
import torch

n_features=2
hidden_dim=5

debug=1
debug_class=0

def printDbg(*argv):
    if debug:
        print("DBG:", end=" ")
        for arg in argv:
           print(arg, end=" ")
        print("\n")

def printFnc(func):
    print("printFnc: func: ", func)
    def inner(*args):
        #print("printFcn.inner() entered")
        if debug_class:
            print("func: ", func.__name__, end= ' ')
            print("args: ", end=' ')
            for i in args[1:]:
                print(i, end=' ')
            print("\n")
        return func(*args)
    return inner

def get_variable_name(obj, namespace):
    print("get_variable_name: namespace.items(): ", namespace.items())
    return [name for name, value in namespace.items() if value is obj][0]

@printFnc
def namestr(obj, namespace):
    DEBUG=0

    if DEBUG:
        print("namestr entered...")
        print("obj: ", obj, type(obj))
        print("namespace: ", type(namespace))

    if type(namespace) == dict:

        if DEBUG:
            for key in namespace.keys():
                print(" - ", key, ": ", namespace[key])
        return [name for name in namespace if namespace[name] is obj]
    else:
        # This will not work, only dict type work. Keeping herr just in case or delete after a while.

        for i in namespace:
            if DEBUG:
                print("id(obj): ", id(obj), obj)
                print(" - ", i, id(i))
            if  getattr(i) == obj:
                return i

'''
Because it is impossible to get class member names using globals() call when using printTensor(<varName>, globals)
when it is called within a class to print member name, getGlobalsClass() will be substituted for globals() class
which returns the same dict object as globals().
so when used inside a class, use following calling to get the same result as regular variable:
printTensor(<class_member_var>, getGlobalsClass(<class_instance>)

i.e.
class m:
    n = 1
    def printClassMember(self):
        printTensor(m, getGlobalsClass(self)

m1=m()
m1=printClassMember()
'''

def getGlobalsClass(pObj):
    DEBUG=0

    if DEBUG:
        print("getGlobalClass entered...")
    d1={}
    for i in dir(pObj):
        d1[i] = getattr(pObj,i)

    if DEBUG:
        print("returning: ", type(d1))
        for i in d1.keys():
            print (" -- ", i, d1[i])

    return d1

#   Prints tensor values in a uniform format
#   Usage example:
#   In order to display the variable name (precedes the value) the pGlobals variable used.
#   if it is called from __main__ function from python, use following calling convention:
#   1) printTensor(<var_name>, globals()), where globals will pass the dictionary object containing
#   current symbol table.
#   If used within a module other than __main__, use locals() call instead:
#   2) printTensor(<var_name>, locals()), where similar dictionaries will be passed from local module scope.
#   If used to call from class member function to print class member variables, use following function calls:
#   3) printTensor(<var_name>, getGlobalClass()) where getGlobalClass will build dictionary from symbols within a 
#   class scope. 

def printTensor(pVar, pGlobals=None, pOverride=None):
    DEBUG=0
    if DEBUG:
        print("printTensor entered...")

    CONFIG_PRINT_TENSOR_SHAPE_ONLY=1
    CONFIG_PRINT_TENSOR_BRIEF=0
    CONFIG_PRINT_TENSOR_BRIEF_DIM1=4
    CONFIG_PRINT_TENSOR_BRIEF_DIM2=3
    CONFIG_PRINT_TENSOR_BRIEF_DIM3=1

    g=None
    if pGlobals:
        if DEBUG:
            print("globals: OK")
            for i in pGlobals.keys():
                print(" - ", i, pGlobals[i])
        g=pGlobals
    else:
        if DEBUG:
            print("globals: None.")
        g=globals()
    # for debug purpose only.
    #for i in g:
    #    print(i)

    lVar=pVar
    lVarName=None

    if type(pVar)==torch.utils.data.dataloader.DataLoader:
        dim1=len(pVar.dataset)
        dim2_0=numpy.array((pVar.dataset)[0][0]).shape
        dim2_1=numpy.array((pVar.dataset)[0][1]).shape

        # force shape only

        pOverride="shape"

    if type(pVar)==list:
        lVar=numpy.array(pVar)

    if type(pVar)==int:
        lVar=numpy.array([pVar])

    if lVarName==None:
        lVarName=namestr(pVar,g)

    print('--------------------------------')
    if pOverride == "full":
        CONFIG_PRINT_TENSOR_SHAPE_ONLY=0
        CONFIG_PRINT_TENSOR_BRIEF=0
    elif pOverride == "brief":
        CONFIG_PRINT_TENSOR_SHAPE_ONLY=0
        CONFIG_PRINT_TENSOR_BRIEF=1
    elif pOverride == "shape":
        CONFIG_PRINT_TENSOR_SHAPE_ONLY=1
        CONFIG_PRINT_TENSOR_BRIEF=0
    elif pOverride == None:
        pass
    else:
        print("Warning: unknown pOverride value while it is not None(default): ", pOverride)
        pass

    if CONFIG_PRINT_TENSOR_SHAPE_ONLY: 

        # special care for dataloader
        if type(pVar)==torch.utils.data.dataloader.DataLoader:
            print(lVarName, ": ", type(pVar), ", [", dim1, ", [", dim2_0, ", ", dim2_1, "]]")
        else:
            print(lVarName, ": ", type(pVar), lVar.shape)
    elif CONFIG_PRINT_TENSOR_BRIEF:
        print(lVarName, ": ", type(pVar), lVar.shape)
        pLenDim1 = min(len(lVar), CONFIG_PRINT_TENSOR_BRIEF_DIM1)
        pLenDim2 = None
        pLenDim3 = None

        try:
            pLenDim2 = min(len(lVar[0]), CONFIG_PRINT_TENSOR_BRIEF_DIM2)
        except Exception as msg:
            pass

        try:
            pLenDim3 = min(len(lVar[0][0]), CONFIG_PRINT_TENSOR_BRIEF_DIM3)
        except Exception as msg:
            pass

        if DEBUG:
            print("pLenDims:", pLenDim1, pLenDim2, pLenDim3)

        if pLenDim3:
            print([d2[0:pLenDim2] for d2 in lVar[0:pLenDim1]])
        elif pLenDim2:
            # does not crash but does not appears working..
            # test well.
            # with array in ch8-p96-rnn-cell.py: p254,4,2 seems to print last dimension as 2 instead of 1 when pLenDim1 is calculated to be 1.
            print( [d3[0:pLenDim3] for d3 in [d2[0:pLenDim2] for d2 in lVar[0:pLenDim1]]  ])
        else:
            # all other cases, print only 1st dimension.
            print(lVar[0:pLenDim1])

    else:
        print(lVarName, ": ", type(pVar))
        print(lVar.shape)
        print(lVar)

    print('--------------------------------')

