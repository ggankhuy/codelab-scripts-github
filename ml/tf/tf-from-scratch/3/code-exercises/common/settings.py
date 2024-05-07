import inspect

n_features=2
hidden_dim=5

debug=0
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

def printTensor(pVarName, pGlobals=None):
    print('--------------------------------')
    g=None
    if pGlobals:
        g=pGlobals
    else:
        g=globals()
    print("globals: ")
    #for i in g:
    #    print(i)
    #print(namestr(pVarName, globals()))
    print(namestr(pVarName, g))
    print('--------------------------------')

'''
@printFnc
def printTensor(pVarName):
    print('--------------------------------')
    #custom_variable_name = get_variable_name(pVarName, locals())
    #print(custom_variable_name, ": ")
    print(namestr(pVarName, globals()))
    print(pVarName, ": ", type(pVarName))
    print(pVarName.shape)
    print(pVarName)
    print('--------------------------------')
'''
