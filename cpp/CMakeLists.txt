# RoboSim C++ extension — pybind11 module exposing hot-path solvers.
#
# Builds a single shared object ``robosim_cpp`` that lands inside the
# Python package (next to the existing pure-Python modules) so the
# rest of the codebase can do ``from robosim import _cpp`` without
# environment-variable juggling.

cmake_minimum_required(VERSION 3.18)
project(robosim_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Honour scikit-build-core's compiler/toolchain choices and any -DCMAKE_*
# the parent build system has set; only fall back to sensible defaults
# when the user didn't specify anything.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Wheel-friendly: don't write absolute build-host paths into the dylib.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# ── Find dependencies ──────────────────────────────────────────────────────
# scikit-build-core puts the right Python interpreter on the path so
# pybind11's ``find_package`` picks it up automatically.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

message(STATUS "Python ${Python_VERSION}: ${Python_EXECUTABLE}")
message(STATUS "pybind11 ${pybind11_VERSION}")
message(STATUS "Eigen3 ${Eigen3_VERSION}: ${EIGEN3_INCLUDE_DIRS}")

# ── Extension target ───────────────────────────────────────────────────────
pybind11_add_module(_cpp MODULE
    bindings/module.cpp
    bindings/transform.cpp
    bindings/kinematics.cpp
    bindings/rbd.cpp
    bindings/cb.cpp
    src/kinematics.cpp
    src/rbd.cpp
    src/cb.cpp
)

target_include_directories(_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(_cpp PRIVATE
    Eigen3::Eigen
)

# Optimisation flags — kept conservative so the build behaves the same
# across reviewers. ``-march=native`` is host-specific so it stays off
# for distributable wheels; the parent build script (or developer) can
# re-enable via ``-DCMAKE_CXX_FLAGS_RELEASE`` for local benchmarking.
target_compile_options(_cpp PRIVATE
    $<$<CONFIG:Release>:-O3>
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# Land the .so inside robosim/ so ``from robosim import _cpp`` works.
install(TARGETS _cpp LIBRARY DESTINATION robosim)
