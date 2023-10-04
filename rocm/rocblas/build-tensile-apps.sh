# build success with following environment:
# rocm5.2 installed (rocm5.4 was OK)
# Tensile source code in ~/gg/git/, once cloned checkout using tensile tag (from 5.2): 9ca08f38c4c3bfe6dfa02233637e7e3758c7b6db
# Checkout using tensile_tag with later than 5.2 will break (i.e.5.4).
# 
FILENAME=tensile_dat_reader
TENSILE_ROOT=/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile/Tensile/Source/lib/include/
TENSILE_ROOT=/root/gg/git/Tensile
TENSILE_ROOT=/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile
TENSILE_SRC_ROOT=$TENSILE_ROOT/Tensile/Source/lib/source/

for dbg in "" "-g"  ; do
    if [[ $dbg -eq "-g" ]]; then
        dbgsuffix="dbg"
    fi

    # compile stage. 

    hipcc  -std=c++17 -I$TENSILE_ROOT/Tensile/Source/lib/include/  -c \
        $FILENAME.cpp \
        $TENSILE_SRC_ROOT/Tensile.cpp \
        $TENSILE_SRC_ROOT/Debug.cpp \
        $TENSILE_SRC_ROOT/msgpack/MessagePack.cpp \
        $TENSILE_SRC_ROOT/ContractionSolution.cpp \
        $TENSILE_SRC_ROOT/ContractionProblem.cpp \
        $TENSILE_SRC_ROOT/TensorOps.cpp \
        $TENSILE_SRC_ROOT/DataTypes.cpp \
        $TENSILE_SRC_ROOT/AMDGPU.cpp  \
        $TENSILE_SRC_ROOT/Utils.cpp  \
        $TENSILE_SRC_ROOT/KernelArguments.cpp \
        $TENSILE_SRC_ROOT/ArithmeticUnitTypes.cpp \
        $TENSILE_SRC_ROOT/EmbeddedData.cpp \
        $TENSILE_SRC_ROOT/EmbeddedLibrary.cpp \
        $TENSILE_SRC_ROOT/KernelLanguageTypes.cpp \
        $TENSILE_SRC_ROOT/PerformanceMetricTypes.cpp \
        $TENSILE_SRC_ROOT/ScalarValueTypes.cpp \
        $TENSILE_SRC_ROOT/TensorDescriptor.cpp

    # linker stage.
    
    hipcc \
        $FILENAME.o \
        Tensile.o  \
        Debug.o \
        MessagePack.o \
        ContractionSolution.o \
        ContractionProblem.o \
        TensorOps.o \
        DataTypes.o \
        Utils.o \
        AMDGPU.o \
        KernelArguments.o \
        ArithmeticUnitTypes.o \
        EmbeddedData.o \
        EmbeddedLibrary.o \
        KernelLanguageTypes.o \
        PerformanceMetricTypes.o \
        ScalarValueTypes.o \
        TensorDescriptor.o \
        -o /opt/rocm/lib/librocblas.so -o $FILENAME.$dbgsuffix.out
done
