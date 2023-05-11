# patch for all rocm release.

# patches related to rocblas.

function patch_rocblas() {
    echo "patch_rocblas: entered..."
    echo "Copying to $1..."
    cat $2/artifacts/rocblas/virtualenv.cmake > $1/virtualenv.cmake
}


