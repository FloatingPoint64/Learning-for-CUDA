cmake_minimum_required(VERSION 3.0.0)
project(reduction VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

# find cuda packages
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("CUDA Version: " ${CUDA_VERSION})
endif()

# include directory
include_directories($CUDA_INCLUDE_DIRS)

include_directories(./)
cuda_include_directories(./gpu)

# set CUDA options
set(CMAKE_CXX_FLAGS "-O1 -Wall /MD")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g /MDd")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG /MD")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG /MD")

set(CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")

#set(CUDA_SEPARABLE_COMPILATION ON)

# sources
file(GLOB CPU_SOURCES "*.cpp")
file(GLOB GPU_SOURCES "gpu/*.cu")

if(MSVC)
    #string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    #string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    #string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    #string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    #string(REPLACE "/MD" "/MT" CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
    #string(REPLACE "/MD" "/MT" CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG})
    #string(REPLACE "/MD" "/MT" CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE})
    #string(REPLACE "/MD" "/MT" CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO})
endif(MSVC)

cuda_add_executable(reduction main.cpp ${CPU_SOURCES} ${GPU_SOURCES})

target_compile_features(reduction PRIVATE cxx_range_for)
#target_compile_features(reduction PUBLIC cxx_range_for)

target_link_libraries(reduction "${CUDA_LIBRARIES}")
