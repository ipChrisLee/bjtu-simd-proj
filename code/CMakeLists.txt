cmake_minimum_required(VERSION 3.29)
project(bjtu-simd-proj)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(clib-std)
target_include_directories(clib-std PUBLIC clib/include)
target_sources(clib-std PUBLIC clib/src-std/tensor.c)

add_library(clib-simd)
target_include_directories(clib-simd PUBLIC clib/include)
target_sources(clib-simd PUBLIC clib/src-simd/tensor.c)

add_library(tlib)
target_include_directories(tlib PUBLIC tlib/include)
target_include_directories(tlib PUBLIC clib/include)
target_sources(tlib PUBLIC tlib/src/test-helper.c tlib/src/test-layer.c)

add_executable(alex-net)
target_sources(alex-net PUBLIC src/main.c)