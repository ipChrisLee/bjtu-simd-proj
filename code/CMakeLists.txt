cmake_minimum_required(VERSION 3.29)
project(bjtu-simd-proj C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

link_libraries(m)

add_library(clib-std)
target_include_directories(clib-std PUBLIC clib/include)
target_sources(clib-std PUBLIC clib/src-std/tensor.c clib/src-common/see.c)

add_library(clib-simd)
target_include_directories(clib-simd PUBLIC clib/include)
target_sources(clib-simd PUBLIC clib/src-simd/tensor.c clib/src-common/see.c)

add_library(tlib)
target_include_directories(tlib PUBLIC tlib/include)
target_include_directories(tlib PUBLIC clib/include)
target_sources(tlib PUBLIC tlib/src/test-helper.c tlib/src/test-layer.c)
target_link_libraries(tlib PUBLIC clib-std)

add_executable(run-layer-test-std)
target_sources(run-layer-test-std PRIVATE exe/run-layer-test.c)
target_link_libraries(run-layer-test-std PUBLIC tlib clib-std)

add_executable(alex-net-std)
target_sources(alex-net-std PRIVATE exe/alex-net.c)
target_link_libraries(alex-net-std PUBLIC clib-std)