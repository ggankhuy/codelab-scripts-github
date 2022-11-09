import re
import time
import sys
import os
import subprocess
from matplotlib import pyplot as plt
import networkx as nx

DEBUG=1
TEST_MODE=1
DEBUG_L2=0
components_built=[]
graph = nx.DiGraph()

# used by def recur_pred

all_pred=[]
indent=""

# Enable directed aclyctic graph implementation of depdendencies wip.

CONFIG_DAG_ENABLE=1 
 
file1=open("graph.dat")
content=file1.readlines()
component=None

try:
    component=sys.argv[1]
except Exception as msg:
    print(msg)
    print("Component not specified, will build everything.")

#   recurring predecessor to find all ancentral predecessors from current node.
#   buildDag must have been called b efore calling this function to populate graph.

def recur_pred(lNode, indent):
    print(indent, "recur_pred: ", lNode)
    preds=list(graph.predecessors(lNode))
    print(indent, "predecessors returned for ", lNode, ": ", preds)
    indent+="  "
    for i in preds:
        recur_pred(i, indent)

        if not i in all_pred:
            print("adding ", i, " to all_pred")
            all_pred.append(i)
        else:
            print(i, " is already in all_pred list, bypassing.")

def buildDag(content):
    # we havent implemented partial dag base on component specified.
    if DEBUG:
        print("---------------------")
        print("buildDag entered: p1: ")
    found=0

    if not content:
        print("File is not read.")
        exit(1)

    for i in content:
        if DEBUG:
            print("................")
        if i == "":
            if DEBUG:
                print("empty line...")
        else:
            #rocBLAS: hipAMD -> rocBLAS child, hipAMD parent. 
            child=i.split(':')[0].strip()
            parents=i.split(':')[1].strip().split(' ')
            for i in range(0, len(parents)):
                parents[i].strip()
    
            if DEBUG:            
                print("child: ", child)
                print("parent: ", parents)

            for i in parents:
                if i:
                    if DEBUG:
                        print("Adding parent: ", i, "child: ", child)
                    graph.add_edges_from([(i.strip(), child)])
                else:
                    if DEBUG:
                        print("empty parent, bypassing...")

    if DEBUG_L2:
        print(graph.nodes())
    list_dag=list(nx.topological_sort(graph))

    if DEBUG:
        print("sorted list: ", list_dag)
        print("buidlDag: done..")
        print("--------------")

    return list_dag

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
    finalList=[]

    print("Reading list.dat")
    file2=open("list.dat")
    content2=file2.readlines()
    list_dag=buildDag(content)
    list_non_dag=[]

    for i in content2:
        i=i.strip()
        if re.search("\#", i):
            if DEBUG:
                print("- Bypassing commented line:", i)
            continue   
        if i in list_dag:
            if DEBUG:
                print("-", i, " is in DAG list, bypassing...")
        else:
            if DEBUG:
                print("- adding ", i)
                list_non_dag.append(i)

    # At this stage, both lists are complete and separate.

    # logic:
    # if component:
        # component in graph.dat
            # build dag and build everything in dag
        # else (component in list.dag
            # only build component.
    # else (not component):
        # build dag and build everything in dag.

    if component:
        if DEBUG:
            print("component specified: ", component)
        if component in list_dag:
            recur_pred(component, indent)
            finalList=all_pred + [component]
        elif component in content1: 
            finalList.append(component)
    else:
        finalList=list_dag + list_non_dag

    print("Final list: ", finalList)
    
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
