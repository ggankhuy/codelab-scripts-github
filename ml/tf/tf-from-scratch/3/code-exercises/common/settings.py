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
    g=None
    if pGlobals:
        g=pGlobals
    else:
        g=globals()

    # for debug purpose only.
    #for i in g:
    #    print(i)

    varName=pVarName

    if type(pVarName==list):
        varName=numpy.array(pVarName)

    print('--------------------------------')
    if pOverride == "full":
        CONFIG_PRINT_TENSOR_SHAPE_ONLY=0
    elif pOverride == "brief":
        CONFIG_PRINT_TENSOR_SHAPE_ONLY=1
    else:
        pass
    if not CONFIG_PRINT_TENSOR_SHAPE_ONLY: 
        print(namestr(varName,g), ": ", type(varName))
        print(varName.shape)
        print(varName)
    else:
        print(namestr(varName,g), ": ", type(varName), varName.shape)

    print('--------------------------------')

