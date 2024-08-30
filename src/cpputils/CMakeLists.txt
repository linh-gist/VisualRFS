cmake_minimum_required (VERSION 3.3)
project (cpputils)

add_subdirectory(pybind11)

##### Murty
#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
SET( EIGEN3_INCLUDE_DIR "/media/ubuntu/2715608D71CBF6FC/eigen-3.4.0" )
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

##### Fuzzy
include_directories (./fuzzylite/fuzzylite)
link_directories(./fuzzylite/fuzzylite/release/bin)

##### OpenMP
# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

pybind11_add_module(cpputils src/main.cpp)
target_link_libraries(cpputils PRIVATE  fuzzylite-static)