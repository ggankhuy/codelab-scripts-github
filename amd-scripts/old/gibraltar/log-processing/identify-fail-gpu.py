# Gather gpu inventory.
# For each gpu, regex search for error pattern.
# Print failing gpu list
# Print unaffected gpu list.
# Make sure they are exclusive.

#    p1 - file name to search
#    p2 - error pattern to search.
#    p3 - gim-legacy, libgv or both.

import sys
import os
import re
import platform
import time

DEBUG=1
CONFIG_INIT_TYPE_GIM_INIT=1
CONFIG_INIT_TYPE_LIBGV_INIT=2
CONFIG_INIT_TYPE_BOTH_INIT=3 # used if log contains both libgv and gim.
CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_BOTH_INIT
fileName=None
pattern=None
fp=None
gpu_list_all=[]
gpu_list_fail=[]
CONFIG_INIT_TYPE_GIM_INIT=1
CONFIG_INIT_TYPE_LIBGV_INIT=2
CONFIG_INIT_TYPE_BOTH_INIT=3 # used if log contains both libgv and gim.

CONFIG_OS=platform.platform()
if re.search("Linux", CONFIG_OS):
    dir_delim="/"
elif re.search("Windows", CONFIG_OS):
    dir_delim="\\"
else:
    print("Unknown OS, can not continue.")
    quit(1)

try:
    if sys.argv[1] == "--help":
            print("*********************************************************")
            print("Gathers gpu inventory. For each gpu search for error pattern during libgv init.")
            print("Collapses the multiple occurrences/flooding of error messages per gpu.")
            print("*********************************************************")
            print("Usage: ", sys.argv[0], " file=<filename to be bisected>, pattern=<error pattern to be searched> init=<gim type>: either libgv or gim or both ")
            print("*********************************************************")
            exit(0)
except Exception as msg:
    print("Continuing...")

for i in sys.argv:
    print("---- Processing---- ", i)
    try:
        if re.search("file=", str(i)):
            fileName=i.split('=')[1]
            print("Found filename to be opened: ", fileName)
			
        if re.search("ooo=", str(i)):
            if (i.split('=')[1] == "yes"):
                print("Out of order log bisect is specified.")			

        if re.search("pattern=", str(i)):
            pattern=i.split('=')[1]
            print("Found error pattern to be used", pattern)

        if re.search("init=", str(i)):
            if i.split('=')[1] == "libgv":
                print("Libgv selected.")
                CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_LIBGV_INIT
            elif i.split("=")[1] == "gim":
                print("gim selected.")
                CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_GIM_INIT
            elif i.split("=")[1] == "both":
                CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_BOTH_INIT
            else:
                print("Invalid init option, choose either 'gim' or 'libgv':", i)
                exit(1)
                            
    except Exception as msg:
        print(msg)
        print("No argument provided")
        print("Assuming init type is libgv...")
        CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_LIBGV_INIT

if fileName==None:
    print("Did not find filename! Use file=<filename> to specify file to be bisected.")
    exit(1)
if pattern==None:
    print("Err. pattern needs to be specified.")
    exit(1)

if CONFIG_INIT_TYPE==CONFIG_INIT_TYPE_LIBGV_INIT:
    gim_init_delimiter="Start AMD open source GIM initialization"
    gpu_init_delimiter="AMD GIM start to probe device"    
    gpu_found_delimiter="AMD GIM probed GPU"
    gpu_search_delimeter="\[[0-9a-f]+:[0-9a-f]+:[0-9]\]"
elif CONFIG_INIT_TYPE==CONFIG_INIT_TYPE_GIM_INIT:
    gim_init_delimiter="AMD GIM init"
    gpu_init_delimiter="SRIOV is supported"    
    gpu_found_delimiter="found:"
    gpu_search_delimeter="[0-9a-f]+:[0-9a-f]+\.[0-9]"
elif CONFIG_INIT_TYPE==CONFIG_INIT_TYPE_BOTH_INIT:
    gim_init_delimiter="Start AMD open source GIM initialization|AMD GIM init"
    gpu_init_delimiter="AMD GIM start to probe device|SRIOV is supported"    
    gpu_found_delimiter="AMD GIM probed GPU|found:"
    gpu_search_delimeter="\[[0-9a-f]+:[0-9a-f]+:[0-9]\]"
else:
    print("Invalid init option, choose either 'gim' or 'libgv':", i)
    exit(1)

print("Delimiters: ", gim_init_delimiter, ", ", gpu_init_delimiter)
print("If these delimiter string changes in future version of libgv, this script may break. Check often the gim init and gpu initialization log periodically.")


# Gather gpu data.

try:
    fp=open(fileName)
except Exception as msg:
    print(msg)
    exit(1)

fp_content=fp.readlines()

if not fp_content:
    print("Failed to read file...")
    exit(1)

for i in fp_content:
    if re.search(gpu_found_delimiter, i):
        if DEBUG:
            print("Found gpu: ", i)
        gpu_list_all.append(re.sub("0000:", "", i.strip().split()[-1]))

    if re.search(pattern, i):
        print(i)
        # search for BDF:
        m=re.search(gpu_search_delimeter, i)
        if m:
            if DEBUG:
                print("expression matched: ", m.group(0)) 
            time.sleep(0)
            result=re.sub("\]|\[","", m.group())
            if result:
                if result in gpu_list_fail:
                    if DEBUG:
                        print("Warn: ", result, " is already in list")
                else:
                    if DEBUG:
                        print("Adding ", result)
                    gpu_list_fail.append(result)
        else:
            if DEBUG:
                print("Expression not matched.")
        
print("GPU inventory: (" + str(len(gpu_list_all)) + ")")
gpu_list_all.sort()
gpu_list_fail.sort()
for i in gpu_list_all:
    print(i)

print("failed GPU inventory: (" + str(len(gpu_list_fail)) + ")")
for i in gpu_list_fail:
    print(i)
