#https://stackoverflow.com/questions/65886495/how-to-link-so-files-with-cmake

cmake_minimum_required(VERSION 3.10)

SET(CMAKE_C_COMPILER "/usr/bin/hipcc")
SET(CMAKE_CXX_COMPILER "/usr/bin/hipcc")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

set(PROJECT_NAME vector)

message(STATUS projectname...: ${PROJECT_NAME})
message(STATUS subprojectname: ${SUB_PROJECT_NAME})

include_directories("/opt/rocm/hip/include")
project(${PROJECT_NAME})
configure_file(vector.h.in ../vector.h)
configure_file(vector.h.in vector.h)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_compile_definitions(${PROJECT_NAME} PUBLIC OPT_SUB_PROJECT_NAME=${SUB_PROJECT_NAME})
if (OPT_SUB_PROJECT_NAME)
    message(STATUS "libamdhip props: ")
else()
    message(STATUS "libamdhip propsx: ")
endif()

list(APPEND CMAKE_PREFIX_PATH)

# does not work without building hip first...
#find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} $ENV{ROCM_PATH})
#find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )

add_library(externalLibA SHARED IMPORTED)
set_target_properties(externalLibA PROPERTIES IMPORTED_LOCATION ${ROCM_PATH}/hip/lib/libamdhip64.so)
get_target_property(DBG_LIBAMDHIP externalLibA IMPORTED_LOCATION)
message(STATUS "libamdhip props: ${DBG_LIBAMDHIP}")
target_include_directories(externalLibA INTERFACE ${ROCM_PATH}/lib)
target_link_libraries(${PROJECT_NAME} PUBLIC externalLibA)

