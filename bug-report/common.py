debug=0
debug_info=0

COL_NAME_PRIORITY="PRIORITY"
COL_NAME_TYPE="TYPE"
COL_NAME_ISSUE_ID="ISSUE_ID"
COL_NAME_STATUS="STATUS"
COL_NAME_TITLE="TITLE"
COL_NAME_CREATED_TIME="CREATED_TIME (UTC)"
COL_NAME_MODIFIED_TIME="MODIFIED_TIME (UTC)"

listColumns=[\
	COL_NAME_PRIORITY, COL_NAME_TYPE, COL_NAME_ISSUE_ID, \
	COL_NAME_STATUS, COL_NAME_TITLE, COL_NAME_CREATED_TIME, COL_NAME_MODIFIED_TIME]

def printBarSingle():
	print("-----------------------------------------------")

#	Given headers, populate the dictionary type column based on header read from csv file.
#	Input:
#		pHeaders: header read from csv file (1st row)
#		pListExclude <list>: name of the column(s) to exclude from header.
#	Output:
#		<dict> - dictionary object with keys column name and values column indexes in headers.
#		None - for any errors.

def setColumnIndices(pHeaders, pListExclude=[]):
	debug=0
	COL_PRIORITY=1
	COL_TYPE=2
	COL_ISSUE_ID=6
	COL_STATUS=5
	COL_CREATE_DATETIME=6
	COL_MODIFY_DATETIME=7
	COL_TITLE=3

	COL_INDICES={\
	"PRIORITY": COL_PRIORITY, \
	"TYPE": COL_TYPE, \
	COL_NAME_ISSUE_ID: COL_ISSUE_ID, \
	COL_NAME_STATUS: COL_STATUS, \
	COL_NAME_CREATED_TIME: COL_CREATE_DATETIME, \
	COL_NAME_MODIFIED_TIME: COL_MODIFY_DATETIME, \
	COL_NAME_TITLE: COL_TITLE \
	}

	
	for i in range(0, len(COL_INDICES)):
		keys=list(COL_INDICES.keys())
		values=list(COL_INDICES.values())
		
		if debug:
			print(keys)
			print(values)

		if keys[i] in pListExclude:
			if debug_info:
				print("--- INFO: (setColumnIndices) set to exclude: ", keys[i])
			continue
		
		if not keys[i] in pHeaders:
			print("(setColumnIndices) Error: ", keys[i], " is not in the header")
			print("(setColumnIndices)headers: ", pHeaders)
			return None
		else:
			try:
				values[i] = pHeaders.index(keys[i])
				COL_INDICES[keys[i]] = values[i]
			except Exception as msg:
				print("(setColumnIndices)Fatal error: Can not find the index of ", keys[i], " in headers. ")
				print("(setColumnIndices)headers: ", pHeaders)
				return None
			
			if debug:
				print("(setColumnIndices)Column index of ", keys[i], " is set to ", values[i])

	return COL_INDICES
