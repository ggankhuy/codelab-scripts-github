import inspect
import numpy

n_features=2
hidden_dim=2

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
    return [name for name in namespace if namespace[name] is obj]

def printTensor(pVarName, pGlobals=None, pOverride=None):
    CONFIG_PRINT_TENSOR_SHAPE_ONLY=1
    CONFIG_PRINT_TENSOR_BRIEF=0
    CONFIG_PRINT_TENSOR_BRIEF_DIM1=4
    CONFIG_PRINT_TENSOR_BRIEF_DIM2=3
    CONFIG_PRINT_TENSOR_BRIEF_DIM3=1
    DEBUG=1

    g=None
    if pGlobals:
        g=pGlobals
    else:
        g=globals()

    # for debug purpose only.
    #for i in g:
    #    print(i)

    varName=pVarName


    if type(pVarName)==list:
        varName=numpy.array(pVarName)

    if type(pVarName)==int:
        varName=numpy.array([pVarName])

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
        print(namestr(pVarName,g), ": ", type(varName), varName.shape)
    elif CONFIG_PRINT_TENSOR_BRIEF:
        print(namestr(pVarName,g), ": ", type(varName), varName.shape)
        pLenDim1 = min(len(varName), CONFIG_PRINT_TENSOR_BRIEF_DIM1)
        pLenDim2 = None
        pLenDim3 = None

        try:
            pLenDim2 = min(len(varName[0]), CONFIG_PRINT_TENSOR_BRIEF_DIM2)
        except Exception as msg:
            pass

        try:
            pLenDim3 = min(len(varName[0][0]), CONFIG_PRINT_TENSOR_BRIEF_DIM3)
        except Exception as msg:
            pass

        if DEBUG:
            print("pLenDims:", pLenDim1, pLenDim2, pLenDim3)

        if pLenDim3:
            print([d2[0:pLenDim2] for d2 in varName[0:pLenDim1]])
        elif pLenDim2:
            # does not crash but does not appears working..
            # test well.
            # with array in ch8-p96-rnn-cell.py: p254,4,2 seems to print last dimension as 2 instead of 1 when pLenDim1 is calculated to be 1.
            print( [d3[0:pLenDim3] for d3 in [d2[0:pLenDim2] for d2 in varName[0:pLenDim1]]  ])
        else:
            # all other cases, print only 1st dimension.
            print(varName[0:pLenDim1])

    else:
        print(namestr(pVarName,g), ": ", type(varName))
        print(varName.shape)
        print(varName)

    print('--------------------------------')

