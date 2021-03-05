# Script for scanning the gim/libgv log file and attempt to find matching pattern from log.
# The gim log resulting from issuing certain activity on host or gim is kept as list of pattern.
# The pattern is a dictionary element with 
# key = gpuvsmi command or something similar
# value = matching pattern. The matching pattern can be several lines.

# Matchin dictionary:
# Key 1 = gpuvsmi_reset_gpu
# Value 1 = <libgv log entries of one or more lines resulting from issuing gpuvsmi_reset_gpu>
# Key 2 = gpuvsmi_allocate_vf
# Value 2 = <libgv log entries of one or more lines resulting from issuing gpuvsmi_allocate_vf>
# ...  and so on.

# Because of the variance of factors, matching pattern can not be relied to exact but only approximate.
# To "match" we establish threshold i.e. 80% of the word in dictionary element expected to match.
# The current dictionary element of matching pattern is scanned against the log the same number of lines.
# If threshold is passed, we deduce it is a match and set gpuvsmi or similar activity is candidate.
# If threshold does not pass, we deduce it is not match and loop over next  dictionary element.
# if all the dictionary element passes and nothing match, we move onto next line in the log and start matching.

# Command list: the specific gpuvsmi command is added to this list after pattern match.
# Mismatch list: lines that did not match any pattern in the dictionary values. 

# Actionable code (pseudocode):
# - open log files (libgv or gim)
# - open log files and build dictionary
# - Point to one line past after the line which indicates GIM is initialized.
# - Construct dictionary.
# - start loop:
# - start scanning each line.
# - for each dictionary element
# - - scan and match N-number of lines from libgv log where N is the same as number of line in current dict. value.
# - - if match exceeds threshold, 
# - - - Pick up key and add to command list.
# - - - Move cursor to the line just after the matched lines in the log.
# - - if all the dictionary pattern did not match the with the cursor at current log lines, 
# - - - add this line to mismatch list.
# - - - move cursor one line forward to next line.

import re
import time
import os
import subprocess
import sys
from datetime import datetime

from fuzzywuzzy import fuzz

FILE_NAME_MATCH_STRING="match-string.txt"
FILE_NAME_TARGET=None
MAX_CHAR_PER_LINE=120
DEBUG = 0
THRESHOLD_MIN_TOKEN_SET_RATIO=75
cmds=[]

# fails from cygwin.
#dateString=str(os.popen('date +%Y%m%d-%H-%M-%S').read()).strip() 
now = datetime.now()
dateString=now.strftime("%d%m%Y-%H-%M-%S")
print("date string: ", dateString)
#ret=os.popen('mkdir ' + dateString).read()
os.mkdir(dateString)
os.mkdir(dateString + "/bcompare")

for i in sys.argv:
    print("Processing ", i)
    try:

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

        if re.search("threshold=", str(i)):
            print("String match accuracy threshold specified: ")
            try:
                THRESHOLD_MIN_TOKEN_SET_RATIO=int(i.split('=')[1])
            except Exception as msg:
                printf("Error: threshold needs to be 0-100: ", THRESHOLD_MIN_TOKEN_SET_RATIO)
                quit(1)
                
        if re.search("file=", str(i)):
            fileName=i.split('=')[1]
            print("Found filename to be opened: ", fileName)
            FILE_NAME_TARGET=fileName
                            
        if re.search("ooo", str(i)):
            if (i.split('=')[1] == "yes"):
                print("Out of order log bisect is specified.")
                CONFIG_BISECT_OOO=1
            else:
                print("Out of order log bisect is specified, but it is not yes.")

    except Exception as msg:
        print(msg)
        print("  EXCEPTION: No argument provided")
        print("  EXCEPTION: Assuming init type is libgv...")
        CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_LIBGV_INIT

if FILE_NAME_TARGET: 
    print("Target file: ", FILE_NAME_TARGET)
else:
    print("Target file is not specified.", )
    quit(1)
    
try:
    matchStringBlock=open(FILE_NAME_MATCH_STRING)
    testString=open(FILE_NAME_TARGET)
except Exception as msg:
    print("Failure opening file...")
    print(msg)
    quit(1)
    
    
TEST_MODE=0
CONFIG_EXCLUSE_GIM_INIT=1
lastLineGimInit=0
cursorTestFile=0
counter=0
testStringBlockContent=testString.readlines()
cursorTestFile=len(testStringBlockContent)
print("No. of lines read from testFile: ", cursorTestFile)
testStringBlockContentProcessed=None

if not CONFIG_EXCLUSE_GIM_INIT:
    for i in reversed(testStringBlockContent):
        if re.search("AMD GIM is Running", i):
            print("Last line gim is finished initialized last time in this log: Line No.: ", str(cursorTestFile), str(i))
            lastLineGimInit = cursorTestFile
            break
        cursorTestFile -= 1
        
    testStringBlockContentProcessed=testStringBlockContent[cursorTestFile:]
else:
    testStringBlockContentProcessed=testStringBlockContent

counter = 0
print("Printing first few lines of truncated string block:")
for i in testStringBlockContentProcessed:
    print(i[0:MAX_CHAR_PER_LINE], "...")
    counter += 1
    if counter > 5:
        break

# Construct dictionary.

print("Constructing dictionary for match string blocks.")
dictmatchStringBlock={}

currKey=None
currValue=None

matchStringBlockContent=matchStringBlock.readlines()
print
for i in matchStringBlockContent:
    # If line is debug, then, signal new match string block.

    if DEBUG: 
        print("currLine:", i)
        
    if re.search("DEBUG: ", i):

        # encountered next match string block. Add to the dictionary if it is not first occurrence.
        # Because if it is first occurrence, search just started and nothing to add.

        if currKey and currValue:
            print("currKey: ",currKey)
            dictmatchStringBlock[currKey] = currValue

        # Reset the currValue to empty and currKey to new key found.

        currKey=re.sub("DEBUG: ", "", i).strip()
        if DEBUG:
            print("currKey generated: ", currKey)
        currValue=[]
    else:
    
        # If not DEBUG line means, it is currValue.

        if i.strip():
            currValue.append(i.strip())
            if DEBUG:
                print("currValue so far: ", currValue)

#   Do this for last entry because, the dictionary assignment is at the start of loop code block.

if currKey and currValue:
    if DEBUG:
        print("currKey: ",currKey)
    dictmatchStringBlock[currKey] = currValue        
    
keys=list(dictmatchStringBlock.keys())
values=list(dictmatchStringBlock.values())

if  DEBUG:
    print("key size/type: ", len(keys), ", ", type(keys))
    print("values size/type: ", len(values), ", ", type(values))

for i in range(0, len(keys)):
    print("i: ", i)
    print("key: ", keys[i])
    if DEBUG:
        print("value: ")
        for j in values[i]:
            print(j)
        print("")

#   Outer loop: Start from test string line by line.
#   Inner loop: iterate through dictionary.
#   for each dict values, determine No. of lines 
#   Grab next No. of lines from test string.

print("Starting match loop...")

#   MatchSet will contain matching entries dictionary to do a bcompare with key + value.
#   TestSet will contain simply the test set to do a compare against.
#   if match fails for block of string, then MatchSet will put the corresponding TestBlock with !!! in front of it.

fileNameMatchSet=dateString + "/bcompare/matchset.log"
fileNameTestSet=dateString + "/bcompare/testset.log"
fpMatchSet=open(fileNameMatchSet, 'w+')
fpTestSet=open(fileNameTestSet, 'w+')

LinesToSkip=0
for cursorTestString in range(0, len(testStringBlockContentProcessed)):

    if TEST_MODE:
        if cursorTestString > 100:
            break

    if LinesToSkip-1 > 0:
        print("Skipping line: ", str(cursorTestString))
        LinesToSkip-=1
        continue
    match_found=None
    print("****** iter: cursorTestString: ", str(cursorTestString), "******")
    currTestBlockFirstLine=testStringBlockContentProcessed[cursorTestString:cursorTestString+2]
    for i in currTestBlockFirstLine:
        print(i)

    currMax=0
    currCmd=""
    for i in list(dictmatchStringBlock.keys()):
        currValue=dictmatchStringBlock[i]

        # Print information.

        print("curr dict key:currValue/len:")
        print(i)

        if DEBUG:
            for j in currValue[0:3]:
                print(j[0:MAX_CHAR_PER_LINE], "...")
            print(len(currValue))
        lines=len(currValue)

        # Construct match block.

        currTestBlock=testStringBlockContentProcessed[cursorTestString:cursorTestString+lines]

        if DEBUG:
            print("currTestBlock/len:")
            for k in currTestBlock[0:2]:
                print("...", k[80:MAX_CHAR_PER_LINE+40], "...")
            print(len(currTestBlock))

        # Concatenate string, both match and test string blocks.

        match_string_concat=' '.join(currValue)
        test_string_concat=' '.join(currTestBlock)

        if DEBUG:
            print("match_string_concat: \n", match_string_concat[0:80])
            print("test_string_concat: \n", test_string_concat[0:80])

        token_set_ratio=fuzz.token_set_ratio(match_string_concat, test_string_concat)
        print("match percent (token_set_ratio): ", str(token_set_ratio))
        
        # If match is above threshold, then move cursor forward in test string same number of lines as current match string block
        # because there is guaranteed to be at least one match. But keep in the loop and and continue updating maximum match.
        # At the end of the loop, maximum match will be assigned to corresponding currCmd command.
        
        if token_set_ratio > THRESHOLD_MIN_TOKEN_SET_RATIO:
            print("Match! : ", str(token_set_ratio))
            match_found=1
            
            if token_set_ratio > currMax:
                currCmd=i
                currMax = token_set_ratio
                currMatchValue = currValue
                currMatchTestBlock = currTestBlock
                print("Current max / command updated to: ", str(currMax), ", ", str(currCmd))
                LinesToSkip=len(currTestBlock)
            
    if match_found:
        print("cmd is: ", str(currCmd))
        cmds.append(currCmd)     
        fpMatchSet.write("------------------\ncmd: " + str(currCmd) + '\n')
        fpTestSet.write("------------------\ncmd: " + str(currCmd) + '\n')
        
        for k in currMatchValue:
            fpMatchSet.write(k + '\n')
        for k in currMatchTestBlock:
            fpTestSet.write(k)
    else:    
        print("Match is not found for line[LineNo:]: ", "[", str(cursorTestString), "]", str(test_string_concat[0:80]))
        cmds.append("? : LineNo: " + str(lastLineGimInit + cursorTestString))
        LinesToSkip=0

        fpMatchSet.write("!!!! " + str(currTestBlockFirstLine[0]))
        fpTestSet.write(str(currTestBlockFirstLine[0]))

fileNameSummary=dateString + "/summary.log"
fileNameDebugLog=dateString + "/debug.log"
fpSummary=open(fileNameSummary, 'w+')
fpDebug=open(fileNameDebugLog, 'w+')

print("*** PRINTING CMDS: ***")
for i in cmds:
    print(i)
    fpSummary.write(i + '\n')

fpSummary.close()
fpDebug.close()

fpMatchSet.close()
fpTestSet.close()
