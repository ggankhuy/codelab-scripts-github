# rocm-build-suite usage (basic)
```
python3 build-rocm.py --help
----------build-rocm.py v1.0
Usage:
--help: display this help menu.
--component=<rocm_component_name>: build specific component. If not specified, builds every rocm component. See a list.dat. 
--dep=<fileName>: specifyc graph file. If not specified, uses default graph file dep.dat
--vermajor=<RocmVersion> force specify rocm version. i.e. 5.2
--verminor=<RocmMinorVersion> force specify rocm minor version. If not specified, defaults to 0.
--llvmno: Do not build LLVM.
--pyno: Do not build and install python.
--cmakeno: Do not build and install cmake.
--py: Force build and install python.
--cmake: Force build and install cmake.
--package: Build packages whenever possible.
Example:
Build rocBLAS only: python3 build-rocm.py --component=rocBLAS
Build everything:   python3 build-rocm.py
Build hipfft specify gg.dat as dependency file: python3 build-rocm.py --component=hipfft --dep=gg.dat
```
#  Supported Operating systems:

Well-tested on Ubuntu 22.02/20.04 

Minimally tested on Centos 8 Stream. 
  
# rocm-build-suite usage (advanced use)
## Features:

By default, build-rocm automatically downloads the source code (GA release) into current folder and uses it to build. If the source is already downloaded, it will by pass.

Build logs are generated in /log/rocmbuild/. Each component will have its own build log. In additional build-summary.log has summary of build (list) and time it took. 

If rocm is installed, uses /opt/rocm/.info/version to download and build the same version as installed version. Otherwise, specify in command line. 

If you discover certain build requires additional other ROCm to be built, set this in dep.dat and commit to dev branch and perform merge request. i.e. if, hypothetically, a rocFFT build requires additionally a rocSPARSE to be built beforehand but it errors out (because rocSPARSE is not build nor defined as prerequisite in dep.dat), then add following line: "rocFFT: rocSPARSE" or if line leading with rocFFT already exists as "rocFFT: rocBLAS" then add it so that "rocFFT: rocBLAS rocSPARSE". 

If no component specified, entire stack will be build with list maintained in list.dat

# rocm-build-suite internals:
```
build-rocm.py: python based main driver script, calls shell scripts after processing python based variables, process command lines.

sh/common.sh: shell functions to be source by other shell scripts. Called/source prior to build.

sh/prebuild.sh: shell variables to be used everywhere. Called/sourced prior to build.

sh/build.sh: build functions implementations.

sh/earlyinit.sh: setup any steps necessary early in the execution of python script.

list.dat: list of all ROCm component to build.

dep.dat: comma separated list of which component requires prerequisite of any component. Do not change actual code to do any prerequisite build log, instead use this file to maintain dependency. 

```
# Future enhancements (pending):
- debian/fedora based installer package.
- add internal rocm build option.
- add debug build switch support
- add custom install directory than default /opt/rocm/
- architecture specific build: i.e. gfx908
- build packages build option (partially done, not well tested"

# debugging the build tool itself.
In build-rocm.py:

- Set TEST_MODE=1 to debug the script logic itself without building (dry-run)
- Set DEBUG=1 to enable add'l debug outputs.
- Set DEBUG_L2=1 to enable further detailed debug outputs.



