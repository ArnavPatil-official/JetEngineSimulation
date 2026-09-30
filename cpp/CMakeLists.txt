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

# Python bindings: interpreter and pybind11 from .venv (cpp/build.sh passes them).
# The .venv also holds the pip Cantera wheel, whose _cantera.so exports weak
# definitions of Cantera's own C++ symbols. A module linked to the shared conda
# libcantera binds to those and aborts (P8.1 toolchain probe). So Python
# modules link the static libcantera.a, compile with hidden visibility and
# export only their PyInit symbol: the conda copy of Cantera stays private.
if(Python_EXECUTABLE)
  find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  find_library(CANTERA_STATIC libcantera.a PATHS ${CANTERA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
  set(CANTERA_STATIC_DEPS omp sundials_cvodes sundials_idas sundials_nvecserial sundials_core
      sundials_sunlinsollapackdense sundials_sunlinsollapackband hdf5 fmt yaml-cpp lapack blas)
  set(CANTERA_STATIC_DEP_PATHS "")
  foreach(dep ${CANTERA_STATIC_DEPS})
    find_library(DEP_${dep} ${dep} PATHS ${CANTERA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
    list(APPEND CANTERA_STATIC_DEP_PATHS ${DEP_${dep}})
  endforeach()
  add_library(cantera_static_dep INTERFACE)
  target_include_directories(cantera_static_dep SYSTEM INTERFACE ${CANTERA_PREFIX}/include)
  target_link_libraries(cantera_static_dep INTERFACE ${CANTERA_STATIC} ${CANTERA_STATIC_DEP_PATHS})
  target_compile_definitions(cantera_static_dep INTERFACE _LIBCPP_DISABLE_AVAILABILITY)

  # every Python module: static Cantera, hidden symbols, only PyInit exported
  function(catjet_python_module name)
    pybind11_add_module(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE cantera_static_dep)
    set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    target_link_options(${name} PRIVATE "LINKER:-exported_symbol,_PyInit_${name}")
    # Python float arithmetic never fuses multiply-add; neither may the port (G0)
    target_compile_options(${name} PRIVATE -ffp-contract=off)
  endfunction()

  catjet_python_module(catjet_probe bindings/probe.cpp)
  catjet_python_module(catjet_core bindings/catjet_core.cpp catjet_core/v6_engine.cpp
                       catjet_core/enthalpy_turbine.cpp)
endif()

# C++ unit tests (Catch2 from the catjet-cpp env)
find_package(Catch2 3 CONFIG REQUIRED PATHS ${CANTERA_PREFIX} NO_DEFAULT_PATH)
add_executable(catjet_tests tests/test_brentq.cpp)
target_include_directories(catjet_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(catjet_tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(catjet_tests PRIVATE
  CATJET_BRENTQ_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/tests/brentq_reference.txt")
target_compile_options(catjet_tests PRIVATE -ffp-contract=off)
enable_testing()
add_test(NAME catjet_tests COMMAND catjet_tests)
