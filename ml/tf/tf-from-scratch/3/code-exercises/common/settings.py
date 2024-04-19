n_features=2
hidden_dim=2

debug=0
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
        print("func: ", func.__name__, end= ' ')
        print("args: ", end=' ')
        for i in args[1:]:
            print(i, end=' ')
        print("\n")
        return func(*args)
    return inner

