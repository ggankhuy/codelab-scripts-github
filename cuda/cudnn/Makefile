CUDA_PATH ?= /usr/local/cuda/
HOST_COMPILER ?= g++
NVCC	  := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

TARGET_SIZE := 64
# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS	 :=
LDFLAGS	 :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

LIBRARIES += -lcublas
FILENAME=cublas-example1
FILENAME=rnn_example
# Target rules
all: build

build: ${FILENAME}

${FILENAME}.o: ${FILENAME}.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
${FILENAME}: ${FILENAME}.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
#	$(EXEC) mkdir -p ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
#	$(EXEC) cp $@ ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

