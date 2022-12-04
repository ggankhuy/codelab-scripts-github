import re
import time
import sys
import os
import subprocess
#from matplotlib import pyplot as plt
import networkx as nx

DEBUG=1
TEST_MODE=0
DEBUG_L2=0
components_built=[]
graph = nx.DiGraph()
rocmVersionMajor=5.2
rocmVersionMinor=0

# used by def recur_pred

all_pred=[]
indent=""
depFile=None
component=None

# Enable directed aclyctic graph implementation of depdendencies wip.

CONFIG_DAG_ENABLE=1 

def dispHelp():
    print("----------build-rocm.py v1.0")
    print("Usage:")
    print("--help: display this help menu.")
    print("--component=<rocm_component_name>: build specific component. If not specified, builds every rocm component.")
    print("--dep=<fileName>: specifyc graph file. If not specified, uses default graph file graph.dat")
    print("--vermajor=<RocmVersion> specify rocm version. i.e. 5.2")
    print("--verminor=<RocmMinorVersion> specify rocm minor version. If not specified, defaults to 0.")
    print("Example:")
    print("Build rocBLAS only: python3 build-rocm.py --component=rocBLAS")
    print("Build everything:   python3 build-rocm.py")
    print("Build hipfft specify gg.dat as dependency file: python3 build-rocm.py --component=hipfft --dep=gg.dat")
    exit(0) 

for i in sys.argv:
    print("Processing ", i)
    try:
        if re.search("--dep=", i):
           depFile=i.split('=')[1].strip()

        if re.search("--component", i):
            component=i.split('=')[1].strip()

        if re.search("--help", i):
            dispHelp()

        if re.search("--vermajor", i):
            rocmVersionMajor=i.split('=')[1].strip()

        if re.search("--verminor", i):
            rocmVersionMinor=i.split('=')[1].strip()

    except Exception as msg:
        print(msg)
        exit(1)

if not component:
    print("Component not specified, will build everything...")
else:
    print("Component: ", component)
   
if not depFile:
    depFile="dep.dat"
    print("dependency file not specified, using default: ", depFile)
else:
    print("dependency file: ", depFile)

depFileHandle=open(depFile)
depFileContent=depFileHandle.readlines()

#   recurring predecessor to find all ancentral predecessors from current node.
#   buildDag must have been called b efore calling this function to populate graph.

def recur_pred(lNode, indent):
    if DEBUG:
        print(indent, "recur_pred: ", lNode)

    preds=list(graph.predecessors(lNode))

    if DEBUG:
        print(indent, "predecessors returned for ", lNode, ": ", preds)

    indent+="  "
    for i in preds:
        recur_pred(i, indent)

        if not i in all_pred:
            if DEBUG:
                print("adding ", i, " to all_pred")

            all_pred.append(i)
        else:
            print(i, " is already in all_pred list, bypassing.")

def buildDag(depFileContent):
    # we havent implemented partial dag base on component specified.
    if DEBUG:
        print("---------------------")
        print("buildDag entered: p1: ")
    found=0

    if not depFileContent:
        print("File is not read.")
        exit(1)

    for i in depFileContent:
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

finalList=[]

if DEBUG:
    print("Reading list.dat")

listDatHandle=open("list.dat")
listDatContent=listDatHandle.readlines()
list_dag=buildDag(depFileContent)
list_non_dag=[]

for i in listDatContent:
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
        if DEBUG:
            print("building partial dag list (recur_pred())")
        recur_pred(component, indent)
        finalList=all_pred + [component]
    elif component in listDatContent:
        print(Component, " you specified is not in depFile. Will build only this component.")
        finalList=[component]
    else:
        print("ERR: Fatal error, it looks like you specified unsupport component.")
        print("Unable to find component: ", component, " in list of component currently supported: ", all_pred)
        exit(1)
else:
    print("building list for everything to build...")
    finalList=list_dag + list_non_dag

print("Final list: ", finalList)
    
counter = 0
for j in finalList:
    if j in components_built:
        print(j + " is already built, bypassing...")
    else:
        if TEST_MODE == 1:
            print("test mode: building " + j)
        else:
            print("calling build script with " + str(j))
            if counter == 0:
                out = subprocess.call(['sh','./sh/build.sh', 'comp=' + str(j), 'vermajor=' + str(rocmVersionMajor), 'verminor=' + str(rocmVersionMinor)])
            else:
                out = subprocess.call(['sh','./sh/build.sh', 'comp=' + str(j), '--llvmno', 'vermajor=' + str(rocmVersionMajor), 'verminor=' + str(rocmVersionMinor)])
            print("out: ", out)

            if out != 0:
                print("Failed to build " + str(out) + ". Unable to continue further.")  
                quit(1)
        components_built.append(j)
        counter += 1

