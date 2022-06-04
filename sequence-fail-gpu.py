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

DEBUG=0
CONFIG_INIT_TYPE_GIM_INIT=1
CONFIG_INIT_TYPE_LIBGV_INIT=2
CONFIG_INIT_TYPE=None
fileName=None
pattern=None
fp=None
gpu_list_all=[]
gpu_list_fail=[]
patterns=None

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
            print("Bisects extra long log from libgv into separate files. Each ")
            print("time gim or libgv is initialized number of times, respective ")
            print("folder is created. ")
            print("Within that, also each gpu initialization within the particular ")
            print("gim initialization log further divided. ")
            print("*********************************************************")
            print("Usage: ", sys.argv[0], " file=<filename to be bisected>, pattern=<pattern>: pattern to search, use | to separate. output=<output_file_name>")
            print("*********************************************************")
            exit(0)
except Exception as msg:
    print("Continuing...")

for i in sys.argv:
    print("Processing ", i)
    try:
        if re.search("file=", i):
            fileName=i.split('=')[1]
            print("Found filename to be opened: ", fileName)

        if re.search("pattern=", i):
            patternsall=i.split('=')[1]
            print("Found error pattern to be used", patternsall)
            patterns=patternsall.split('|')
        
        if re.search("output", i):
            fout=i.split('=')[1]
            
    except Exception as msg:
        print("No argument provided")
        quit(1)

if fileName==None:
    print("Did not find filename! Use file=<filename> to specify file to be bisected.")
    exit(1)
    
if patterns==None:
    print("Err. pattern needs to be specified.")
    exit(1)

if fout==None:
    print("Err. Need to specify output file...")
    exit(1)
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

sequence_errors=[]
last_gpu=""
last_error=""

for i in fp_content:
    if re.search("AMD GIM probed GPU", i):
        if DEBUG:
            print("Found gpu: ", i)
        gpu_list_all.append(re.sub("0000:", "", i.strip().split()[-1]))

    for pattern in patterns:
        if re.search(pattern, i):
        
            # search for BDF:
            
            m=re.search("\[[0-9a-f]+:[0-9a-f]+:[0-9]\]", i)
            if m:
                if DEBUG:
                    print("expression matched: ", m.group(0)) 
                
                result=re.sub("\]|\[","", m.group())
                
                #    Error is found and gpu bdf address is determined. Now determine whether we add or not to avoid duplication lines...
                
                if result:
                    if last_gpu == result and last_error == pattern:                        
                        print("Bypassing since last error and gpu are same")
                    else:
                        print("adding... error|gpu: ", pattern, ", ", result)
                        sequence_errors.append(i)
                        last_gpu=result
                        last_error=pattern
                else:
                    if DEBUG:
                        print("Non matching line")
                
                
                '''
                    if result in gpu_list_fail:
                        if DEBUG:
                            print("Warn: ", result, " is already in list")
                    else:
                        if DEBUG:
                            print("Adding ", result)
                    '''
            else:
                if DEBUG:
                    print("Expression not matched.")
            
fpOut=open(fout,'w')
print("Found lines: ", str(len(sequence_errors)))
time.sleep(3)
for i in sequence_errors:
    print(i)
    fpOut.write(i)
fpOut.close()
        


'''
print("GPU inventory: (" + str(len(gpu_list_all)) + ")")
gpu_list_all.sort()
gpu_list_fail.sort()
for i in gpu_list_all:
    print(i)

print("GPU inventory: (" + str(len(gpu_list_fail)) + ")")
for i in gpu_list_fail:
    print(i)
'''