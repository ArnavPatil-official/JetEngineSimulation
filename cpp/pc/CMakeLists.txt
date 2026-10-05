# Phase 8 PC (Linux/WSL2) build of the same C++ core (docs/plan.md, PC phase).
# Builds the identical cpp/bindings, cpp/catjet_core and cpp/tests sources as
# cpp/CMakeLists.txt into a separate tree (cpp/build_pc, via cpp/build_pc.sh).
# The original cpp/CMakeLists.txt and its Mac-pinned build are never edited;
# this is a new, independent entry point, not a fork of the original file.
cmake_minimum_required(VERSION 3.24)
project(catjet_core_pc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "cpp/pc targets Linux or WSL2 only; native Windows is not supported")
endif()

set(CANTERA_PREFIX "" CACHE PATH "prefix of the PC conda env (libcantera-devel 3.2.0)")
if(NOT CANTERA_PREFIX)
  message(FATAL_ERROR "set -DCANTERA_PREFIX (use cpp/build_pc.sh)")
endif()
find_library(CANTERA_LIB cantera PATHS ${CANTERA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
find_library(FMT_LIB fmt PATHS ${CANTERA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
find_package(OpenMP REQUIRED)

set(CORE "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Flags follow the package's cantera.pc. No fast-math anywhere: G0 is 1e-9.
add_library(cantera_dep INTERFACE)
target_include_directories(cantera_dep SYSTEM INTERFACE ${CANTERA_PREFIX}/include)
target_link_libraries(cantera_dep INTERFACE ${CANTERA_LIB} ${FMT_LIB} OpenMP::OpenMP_CXX)
set(CMAKE_BUILD_RPATH ${CANTERA_PREFIX}/lib)

# P8.1 step 2 twin: HP equilibrium parity with Python Cantera
add_executable(hello_equilibrium ${CORE}/tools/hello_equilibrium.cpp)
target_link_libraries(hello_equilibrium PRIVATE cantera_dep)

# Python bindings: interpreter and pybind11 from .venv (cpp/build_pc.sh passes them).
# As on Mac, the pip Cantera wheel's own C++ symbols must not bind into these
# modules, so Python modules link the static libcantera.a with hidden
# visibility. An anonymous GNU ld version script exports only PyInit_<name>
# and localizes everything else, including symbols from the static archive.
if(Python_EXECUTABLE)
  find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  find_library(CANTERA_STATIC libcantera.a PATHS ${CANTERA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
  set(CANTERA_STATIC_DEPS sundials_cvodes sundials_idas sundials_nvecserial sundials_core
      sundials_sunlinsollapackdense sundials_sunlinsollapackband hdf5 fmt yaml-cpp lapack blas)
  set(CANTERA_STATIC_DEP_PATHS "")
  foreach(dep ${CANTERA_STATIC_DEPS})
    find_library(DEP_${dep} ${dep} PATHS ${CANTERA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
    list(APPEND CANTERA_STATIC_DEP_PATHS ${DEP_${dep}})
  endforeach()
  add_library(cantera_static_dep INTERFACE)
  target_include_directories(cantera_static_dep SYSTEM INTERFACE ${CANTERA_PREFIX}/include)
  target_link_libraries(cantera_static_dep INTERFACE ${CANTERA_STATIC} ${CANTERA_STATIC_DEP_PATHS} OpenMP::OpenMP_CXX)

  # every Python module: static Cantera, hidden symbols, only PyInit exported
  function(catjet_python_module name)
    pybind11_add_module(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE cantera_static_dep)
    set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    set(export_map "${CMAKE_CURRENT_BINARY_DIR}/${name}.exports.map")
    file(GENERATE OUTPUT ${export_map}
      CONTENT "{ global: PyInit_${name}; local: *; };\n")
    target_link_options(${name} PRIVATE "LINKER:--version-script,${export_map}")
    # Python float arithmetic never fuses multiply-add; neither may the port (G0)
    target_compile_options(${name} PRIVATE -ffp-contract=off)
  endfunction()

  catjet_python_module(catjet_probe ${CORE}/bindings/probe.cpp)
  catjet_python_module(catjet_core ${CORE}/bindings/catjet_core.cpp ${CORE}/catjet_core/v6_engine.cpp
                       ${CORE}/catjet_core/enthalpy_turbine.cpp ${CORE}/catjet_core/choking_nozzle.cpp
                       ${CORE}/catjet_core/reactor_network.cpp ${CORE}/catjet_core/maps.cpp
                       ${CORE}/catjet_core/offdesign.cpp)
  # P8.1 benchmark arms 3/4 (std::thread pool; variants a, b, c)
  catjet_python_module(catjet_benchmark ${CORE}/bindings/catjet_benchmark.cpp ${CORE}/catjet_core/v6_engine.cpp
                       ${CORE}/catjet_core/v6_variant.cpp)
endif()

# C++ unit tests (Catch2 from the PC conda env), same sources as the Mac build.
find_package(Catch2 3 CONFIG REQUIRED PATHS ${CANTERA_PREFIX} NO_DEFAULT_PATH)
add_executable(catjet_tests ${CORE}/tests/test_brentq.cpp)
target_include_directories(catjet_tests PRIVATE ${CORE})
target_link_libraries(catjet_tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(catjet_tests PRIVATE
  CATJET_BRENTQ_REFERENCE="${CORE}/tests/brentq_reference.txt")
target_compile_options(catjet_tests PRIVATE -ffp-contract=off)

# P8.3 nozzle contract tests (standalone executable: shared Cantera is fine)
add_executable(catjet_nozzle_tests ${CORE}/tests/test_choking_nozzle.cpp ${CORE}/catjet_core/choking_nozzle.cpp)
target_include_directories(catjet_nozzle_tests PRIVATE ${CORE})
target_link_libraries(catjet_nozzle_tests PRIVATE cantera_dep Catch2::Catch2WithMain)
target_compile_definitions(catjet_nozzle_tests PRIVATE
  CATJET_PERFECT_GAS="${CORE}/tests/fixtures/perfect_gas.yaml"
  CATJET_MECHANISM="${CORE}/../data/creck_c1c16_full.yaml")
target_compile_options(catjet_nozzle_tests PRIVATE -ffp-contract=off)
enable_testing()
add_test(NAME catjet_tests COMMAND catjet_tests)
add_test(NAME catjet_nozzle_tests COMMAND catjet_nozzle_tests)
