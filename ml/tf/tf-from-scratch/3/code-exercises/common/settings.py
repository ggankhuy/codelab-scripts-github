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
    def inner(*args):
        #print("printFcn.inner() entered")
        print("func: ", func.__name__)
        for i in args:
            print("arg: ", i)
    return inner

