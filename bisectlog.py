import sys
gim_init_delimiter="Start AMD open source GIM initialization"
gpu_init_delimiter="AMD GIM start to probe device"

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
	arg1 = sys.argv[1]
except Exception as msg:
	print(msg)
	print("Did not see argv1!!! arg1 should be log file name.")
	exit(1)
	
print("arg1: ", arg1)

try:
	fp=open(arg1)
except Exception as msg:
	print(msg)
	exit(1)
	
fp_content=fp.read()
#print(fp_content)
fp_content_gim_inits=fp_content.split(gim_init_delimiter)

for i in range(0, len(fp_content_gim_inits)):
	fpw=open(arg1 + ".libgv-init-" + str(i) + ".log", "w")
	fpw.write(fp_content_gim_inits[i])
	fpw.close()
	
	fp_content_gpu_inits=fp_content_gim_inits[i].split(gpu_init_delimiter)
	
	for j in range(0, len(fp_content_gpu_inits)):
		fpw1=open(arg1 + ".libgv-init-" + str(i) + "gpu" + str(j) + ".log", "w")
		fpw1.write(fp_content_gpu_inits[j])
		fpw1.close()
		



	
	


