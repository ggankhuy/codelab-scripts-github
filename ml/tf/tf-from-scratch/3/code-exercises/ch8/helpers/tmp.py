a="123"
b="string"

def printVar(pName):
    print("printVar: pName: ", pName)
    items=globals().items()
    for i in items:
        print(type(i),i)
        if i[0].strip() == pName:
            print("match!:", pName)
printVar("a")
printVar("b")
