cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_COMPILER "/opt/gcc/bin/g++-m")
project(cses)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
add_compile_options(-Wall -O3 -march=native -ftree-vectorize -Wno-vla -Wno-unused)
set(SOURCE_FILES sticks.cpp)
add_executable(sticks ${SOURCE_FILES})
