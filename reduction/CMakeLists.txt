cmake_minimum_required(VERSION 3.0.0)
project(reduction VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)

# find cuda packages
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("CUDA Version: " ${CUDA_VERSION})
endif()

# include directory
include_directories($CUDA_INCLUDE_DIRS)
include_directories(./)

# set CUDA options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O2 -gencode arch=compute_61,code=sm_61 --expt-relaxed-constexpr -lineinfo --ptxas-option=-v")

# sources
file(GLOB SOURCES "*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)

#cuda_add_executable(reduction ${SOURCES})
add_executable(reduction ${SOURCES})
