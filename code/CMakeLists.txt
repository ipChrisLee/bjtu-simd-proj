cmake_minimum_required(VERSION 3.29)
project(bjtu-simd-proj)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(clib)
target_include_directories(clib PUBLIC src/clib/include)
target_sources(clib PUBLIC src/clib/src-std/tensor.c)

add_executable(alex-net)
target_sources(alex-net PUBLIC src/main.c)