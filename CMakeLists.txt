cmake_minimum_required(VERSION 3.5)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(Flags INTERFACE)
target_compile_options(Flags INTERFACE
    -Wall
    -Werror
    -march=native
    -std=c++17
    )

add_library(Catch2 INTERFACE)
target_include_directories(Catch2 INTERFACE
    third_party)

add_subdirectory(src)
