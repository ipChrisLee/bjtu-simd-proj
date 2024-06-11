cmake_minimum_required(VERSION 3.29)
project(bjtu-simd-proj)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(clib-std)
target_include_directories(clib-std PUBLIC src/clib/include)
target_sources(clib-std PUBLIC src/clib/src-std/tensor.c)

add_library(clib-simd)
target_include_directories(clib-simd PUBLIC src/clib/include)
target_sources(clib-simd PUBLIC src/clib/src-simd/tensor.c)

add_executable(alex-net)
target_sources(alex-net PUBLIC src/main.c)