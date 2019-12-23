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
include_directories(./gpu)

# set CUDA options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2 -gencode arch=compute_61,code=sm_61 --expt-relaxed-constexpr -lineinfo")

# sources
file(GLOB CPU_SOURCES "*.cpp")
file(GLOB GPU_SOURCES "gpu/*.cu")

# make release version
set(CMAKE_BUILD_TYPE Release)

cuda_add_executable(reduction main.cpp ${CPU_SOURCES} ${GPU_SOURCES})
#add_executable(reduction ${SOURCES})

target_compile_features(reduction PRIVATE cxx_range_for)

target_link_libraries(reduction "${CUDA_LIBRARIES}")
