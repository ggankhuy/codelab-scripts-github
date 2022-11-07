
import re
import time
import sys
import os
import subprocess
from matplotlib import pyplot as plt
import networkx as nx

DEBUG=1
TEST_MODE=1
components_built=[]

# Enable directed aclyctic graph implementation of depdendencies wip.
CONFIG_DAG_ENABLE=1 
 
file1=open("graph.dat")
content=file1.readlines()

try:
    component=sys.argv[1]
except Exception as msg:
    print(msg)
    print("Component not specified, will build everything.")

def buildDag(content):
    print("---------------------")
    print("buildDag entered: p1: ")
    found=0

    if not content:
        print("File is not read.")
        exit(1)

    graph = nx.DiGraph()

    for i in content:
        print("................")
        if i == "":
            print("empty line...")
        else:
            #rocBLAS: hipAMD -> rocBLAS child, hipAMD parent. 
            child=i.split(':')[0].strip()
            parents=i.split(':')[1].strip().split(' ')
            for i in range(0, len(parents)):
                parents[i].strip()
                
            print("child: ", child)
            print("parent: ", parents)

            for i in parents:
                if i:
                    print("Adding parent: ", i, "child: ", child)
                    graph.add_edges_from([(i.strip(), child)])
                else:
                    print("empty parent, bypassing...")

    #print("did not find " + component + " as build target, try building, (could be leaf)...")
    print(graph.nodes())
    print(list(nx.topological_sort(graph)))
    print("buidlDag: done..")
    print("--------------")

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

if CONFIG_DAG_ENABLE:

    buildDag(content)

    '''

    from matplotlib import pyplot as plt
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_edges_from([("root", "a"), ("a", "b"), ("a", "e"), ("b", "c"), ("b", "d"), ("d", "e")])
    print(nx.shortest_path(graph, 'root', 'e'))
    #print(nx.dag_longest_path(graph, 'root', 'e'))
    print(list(nx.topological_sort(graph)))
    '''

    
    exit(0)
findDep(component)
dependencies.reverse()
print("dependencies: ", dependencies)

counter = 0
for j in  dependencies:
    if j in components_built:
        print(j + " is already built, bypassing...")
    else:
        if TEST_MODE == 1:
            print("test mode: building " + j)
        else:
            print("calling build script with " + str(j))
            if counter == 0:
                out = subprocess.call(['sh','./sh/build.sh', 'comp=' + str(j)])
            else:
                out = subprocess.call(['sh','./sh/build.sh', 'comp=' + str(j), '--llvmno'])
            print("out: ", out)

            if out != 0:
                print("Failed to build " + str(out) + ". Unable to continue further.")  
                quit(1)
        components_built.append(j)
        counter += 1

if TEST_MODE:
    print("test mode: building " + str(component))
else:
    out = subprocess.call(['sh','./sh/build.sh', 'comp=' + str(component), '--llvmno'])

#print("Final dependency list:")
#print(dependencies)
