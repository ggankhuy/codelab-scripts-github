# patch for all rocm release.

# patches related to rocblas.

function patch_rocblas() {
    echo "patch_rocblas: entered..."
    echo "Copying to $1..."
    cp $2/artifacts/rocblas/virtualenv.cmake $1/
}


