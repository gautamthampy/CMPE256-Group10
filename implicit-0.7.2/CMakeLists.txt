cmake_minimum_required(VERSION 3.17.0)

# Workaround for MSVC 1950 (Visual Studio 2024/2026) compatibility
# Temporarily override MSVC_VERSION to bypass version checks
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
    set(MSVC_VERSION 1940)
endif()

project(implicit)

enable_testing()

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(OpenMP)

include_directories(.)
set (CMAKE_CXX_STANDARD 11)
add_subdirectory(implicit)
