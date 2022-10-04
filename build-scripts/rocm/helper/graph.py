
import re
import time
import sys
DEBUG=1

file1=open("graph.dat")
content=file1.readlines()
dependencies=[]

try:
    component=sys.argv[1]
except Exception as msg:
    print(msg)
    exit(1)

def findDep(component):
    print("---------------------")
    print("findDep entered: p1: ", component)
#   time.sleep(1)
    for i in content:
        if i == "":
            print("empty line...")
        else:
            if re.findall('^' + component, i):
                print("found the line...")
                deps=i.split(":")[1]
                depsToken=deps.strip().split(' ')
                print("depsToken for ", component, ": ", depsToken)

                for  i in depsToken:
                    if i in dependencies:
                        if DEBUG:
                            print(i, ": already added...")
                    else:
                        if DEBUG:
                            print("adding dep: ", i)
                        dependencies.append(i)
                    findDep(i)
                break


findDep(component)
print("Final dependency list:")
print(dependencies)
