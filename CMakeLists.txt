cmake_minimum_required(VERSION 3.21)
project(benchmark)

set(CMAKE_CXX_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR}/include include/test)

add_compile_options(-DUSE_BRANCH -O3 -fno-tree-vectorize -march=native -mavx2 -mavx)

add_executable(benchmark main.cpp)
