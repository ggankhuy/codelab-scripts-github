
import re
import time
import sys
import os
import subprocess

DEBUG=1
TEST_MODE=0
components_built=[]

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
    found=0

    if not component:
        print("findDep: component is empty, exiting...")
        return 1

    for i in content:
        if i == "":
            print("empty line...")
        else:
            if re.findall('^' + component, i):
                print("found the line...")
                found=1
                deps=i.split(":")[1]
                depsToken=deps.strip().split(' ')
                print("depsToken for ", component, ": ", depsToken)

                for  i in depsToken:
                    if i in dependencies:
                        if DEBUG:
                            print(i, ": already added...")
                    else:
                        if i:
                            if DEBUG:
                                print("adding dep: ", i)
                            dependencies.append(i)
                        else:
                            print("dependency is empty...")
                        findDep(i)
                break
    #print("did not find " + component + " as build target, try building, (could be leaf)...")
    print("findDep.done..")
    print("--------------")

findDep(component)
dependencies.reverse()
print("dependencies: ", dependencies)

for j in  dependencies:
    if j in components_built:
        print(j + " is already built, bypassing...")
    else:
        if TEST_MODE == 1:
            print("test mode: building " + j)
        else:
            print("calling build script with " + str(j))
            out = subprocess.call(['sh','./sh/build.sh', 'comp=' + str(j)])
            print("out: ", out)

            if out != 0:
                print("Failed to build " + str(out) + ". Unable to continue further.")  
                quit(1)
        components_built.append(j)

if TEST_MODE:
    print("test mode: building " + str(component))
else:
    out = subprocess.call(['sh','./sh/build.sh', 'comp=' + str(component)])
#print("Final dependency list:")
#print(dependencies)
