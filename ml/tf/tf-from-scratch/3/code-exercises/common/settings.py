n_features=2
hidden_dim=2

debug=1
def printDbg(*argv):
    if debug:
        print("DBG:", end=" ")
        for arg in argv:
           print(arg, end=" ")

        print("\n")
def printFcn(func):
    def inner():
        print("func.__name__")
    return inner

