cmake_minimum_required(VERSION 3.29)
project(bjtu-simd-proj C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

link_libraries(m)

if(DEFINED ENV{doTimeit})
    add_definitions(-DdoTimeit)
endif()

add_library(clib-std)
target_include_directories(clib-std PUBLIC clib/include)
target_sources(clib-std PUBLIC clib/src-std/tensor.c clib/src-common/see.c)

add_library(tlib-std)
target_include_directories(tlib-std PUBLIC tlib/include)
target_include_directories(tlib-std PUBLIC clib/include)
target_sources(tlib-std PUBLIC tlib/src/test-helper.c tlib/src/test-layer.c)
target_link_libraries(tlib-std PUBLIC clib-std)

add_executable(run-layer-test-std)
target_sources(run-layer-test-std PRIVATE exe/run-layer-test.c)
target_link_libraries(run-layer-test-std PUBLIC tlib-std clib-std)

add_executable(alex-net-std)
target_sources(alex-net-std PRIVATE exe/alex-net.c)
target_link_libraries(alex-net-std PUBLIC clib-std)

add_executable(tensor-diff)
target_sources(tensor-diff PRIVATE exe/tensor-diff.c)
target_link_libraries(tensor-diff PUBLIC clib-std)

if(DEFINED ENV{onQemu} OR DEFINED ENV{onArm} OR DEFINED ENV{onRel}) 
    add_library(clib-simd)
    target_include_directories(clib-simd PUBLIC clib/include)
    target_sources(clib-simd PUBLIC clib/src-simd/tensor.c clib/src-common/see.c)

    add_library(tlib-simd)
    target_include_directories(tlib-simd PUBLIC tlib/include)
    target_include_directories(tlib-simd PUBLIC clib/include)
    target_sources(tlib-simd PUBLIC tlib/src/test-helper.c tlib/src/test-layer.c)
    target_link_libraries(tlib-simd PUBLIC clib-simd)

    add_executable(run-layer-test-simd)
    target_sources(run-layer-test-simd PRIVATE exe/run-layer-test.c)
    target_link_libraries(run-layer-test-simd PUBLIC tlib-simd clib-simd)

    add_executable(alex-net-simd)
    target_sources(alex-net-simd PRIVATE exe/alex-net.c)
    target_link_libraries(alex-net-simd PUBLIC clib-simd)
endif()