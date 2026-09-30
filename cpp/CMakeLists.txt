# Phase 8 C++ core (docs/plan.md P8.1). Build with cpp/build.sh, which pins the
# compiler, SDK and the catjet-cpp conda prefix (registration amendment P8-A1.3).
cmake_minimum_required(VERSION 3.24)
project(catjet_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CANTERA_PREFIX "" CACHE PATH "prefix of the catjet-cpp conda env (libcantera-devel 3.2.0)")
if(NOT CANTERA_PREFIX)
  message(FATAL_ERROR "set -DCANTERA_PREFIX (use cpp/build.sh)")
endif()
find_library(CANTERA_LIB cantera PATHS ${CANTERA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
find_library(FMT_LIB fmt PATHS ${CANTERA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)

# Flags follow the package's cantera.pc. No fast-math anywhere: G0 is 1e-9.
add_library(cantera_dep INTERFACE)
target_include_directories(cantera_dep SYSTEM INTERFACE ${CANTERA_PREFIX}/include)
target_link_libraries(cantera_dep INTERFACE ${CANTERA_LIB} ${FMT_LIB})
target_compile_definitions(cantera_dep INTERFACE _LIBCPP_DISABLE_AVAILABILITY)
set(CMAKE_BUILD_RPATH ${CANTERA_PREFIX}/lib)

# P8.1 step 2: HP equilibrium parity with Python Cantera
add_executable(hello_equilibrium tools/hello_equilibrium.cpp)
target_link_libraries(hello_equilibrium PRIVATE cantera_dep)
