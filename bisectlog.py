import sys
import os
import re

CONFIG_INIT_TYPE_GIM_INIT=1
CONFIG_INIT_TYPE_LIBGV_INIT=2
CONFIG_INIT_TYPE=None
fileName=None

try:
	if sys.argv[1] == "--help":
	print("Usage: ", sys.argv[0] ", file=<filename to be bisected>, " ", init="gim type: either libgv or gim")
	exit(0)
exception as msg:
	print("Continuing...")
	
for i in sys.argv:
	print("Processing ", i)
	try:
		if re.search("init=", i):
			if i.split('=')[1] == "libgv":
				print("Libgv selected.")
				CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_LIBGV_INIT
			elif i.split("=")[1] == "gim":
				print("gim selected.")
				CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_GIM_INIT
			else:
				print("Invalid init option, choose either 'gim' or 'libgv':", i)
				exit(1)
				
		if re.search("file=", i):
			fileName=i.split('=')[1]
			print("Found filename to be opened: ", fileName)
							
	except Exception as msg:
		print("No argument provided")
		print("Assuming init type is libgv...")
		CONFIG_INIT_TYPE=CONFIG_INIT_TYPE_LIBGV_INIT

if fileName==None:
	print("Did not find filename! Use file=<filename> to specify file to be bisected.")
	exit(1)
		
if CONFIG_INIT_TYPE==CONFIG_INIT_TYPE_LIBGV_INIT:
	gim_init_delimiter="Start AMD open source GIM initialization"
	gpu_init_delimiter="AMD GIM start to probe device"
elif CONFIG_INIT_TYPE==CONFIG_INIT_TYPE_GIM_INIT:
	gim_init_delimiter="AMD GIM init"
	gpu_init_delimiter="SRIOV is supported"
else:
	print("Invalid init option, choose either 'gim' or 'libgv':", i)
	exit(1)

print("Delimiters: ", gim_init_delimiter, ", ", gpu_init_delimiter)
print("If these delimiter string changes in future version of libgv, this script may break. Check often the gim init and gpu initialization log periodically.")

# bisects log into different files:
#	each libgv initialization will be separate files.
#	<filename>.libgv-init-0.log
#	<filename>.libgv-init-1.log
#	...
#	<filename>.libgv-init-N.log

# 	each libgv initialization files are also bisected further into each gpu init files.
#	<filename>.libgv-init-0.gpu0.log
#	<filename>.libgv-init-0.gpu1.log
#	...
#	<filename>.libgv-init-0.gpuM.log

#	<filename>.libgv-init-1.gpu0.log
#	<filename>.libgv-init-1.gpu1.log
#	...
#	<filename>.libgv-init-1.gpuM.log

#	<filename>.libgv-init-N.gpu0.log
#	<filename>.libgv-init-N.gpu1.log
#	...
#	<filename>.libgv-init-N.gpuM.log

try:
	fp=open(fileName)
except Exception as msg:
	print(msg)
	exit(1)
	
fp_content=fp.read()
#print(fp_content)
fp_content_gim_inits=fp_content.split(gim_init_delimiter)

for i in range(0, len(fp_content_gim_inits)):
	subdir="libgv-init-" + str(i)
	print("Created subdirectory " + subdir)
	os.mkdir(subdir)
	filename=subdir + "\\" + fileName + ".libgv-init-" + str(i) + ".log"
	print("filename: ", filename)
	fpw=open(filename, "w")
	fpw.write(fp_content_gim_inits[i])
	fpw.close()
	
	fp_content_gpu_inits=fp_content_gim_inits[i].split(gpu_init_delimiter)
	
	for j in range(0, len(fp_content_gpu_inits)):
		fpw1=open(subdir + "\\" + fileName + ".libgv-init-" + str(i) + ".gpu" + str(j) + ".log", "w")
		fpw1.write(fp_content_gpu_inits[j])
		fpw1.close()
		



	
	


