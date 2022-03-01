COMP=hipcc
HIPCC=$(COMP)
EXECUTABLE=p61-roctracer
OBJECTS = p61-roctracer.o
SOURCES=p61-roctracer.cpp
HIP_API_ACTIVITY_ON=1
KFD_API_ACTIVITY_ON=0
ROCTX_API_ACTIVITY_ON=1
HCC_API_ACTIVITY_ON=0
HSA_API_ACTIVITY_ON=1

ROOT_PATH ?= /opt/rocm/roctracer/
INC_PATH  ?= $(ROOT_PATH)/include
LIB_PATH  ?= $(ROOT_PATH)/lib
ROCM_PATH ?= /opt/rocm
ROC_LIBS  = -Wl,--rpath,${LIB_PATH} \
    $(LIB_PATH)/libroctracer64.so \
    $(LIB_PATH)/libroctx64.so \
    $(ROCM_PATH)/lib/libamdhip64.so \
    $(LIB_PATH)/libkfdwrapper64.so \
    $(ROCM_PATH)/rocprofiler/lib/librocprofiler64.so 

FLAGS =-g $(INC_PATH:%=-I%) -I$(ROCM_PATH)/roctracer/include/ -I$(ROCM_PATH)/hsa/include/hsa -I$(ROCM_PATH)/hsa/include -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/include -DLOCAL_BUILD=1 -DHIP_VDI=${HIP_VDI} -DITERATIONS=$(ITERATIONS) -DAMD_INTERNAL_BUILD=1

ifeq ($(HIP_API_ACTIVITY_ON), 1)
		FLAGS += -DHIP_API_ACTIVITY_ON=1
endif
ifeq ($(KFD_API_ACTIVITY_ON), 1)
		FLAGS += -DKFD_API_ACTIVITY_ON=1
endif
ifeq ($(ROCTX_API_ACTIVITY_ON), 1)
		FLAGS += -DROCTX_API_ACTIVITY_ON=1
endif
ifeq ($(HCC_API_ACTIVITY_ON), 1)
		FLAGS += -DHCC_API_ACTIVITY_ON=1
endif
ifeq ($(HSA_API_ACTIVITY_ON), 1)
		FLAGS += -DHSA_API_ACTIVITY_ON=1
endif

all:	$(EXECUTABLE)

$(OBJECTS): $(SOURCES)
	$(COMP) $(FLAGS) -c -o $@ $<

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(ROC_LIBS)

test:  $(EXECUTABLE)
	LD_PRELOAD="$(LIB_PATH)/libkfdwrapper64.so librocprofiler64.so /opt/rocm-4.2.0/lib/libamdhip64.so" ./$(EXECUTABLE) 

testp:  $(EXECUTABLE)
	rocprof --hiptrace ./$(EXECUTABLE) 


clean:
	rm *.o
	rm $(EXECUTABLE)
