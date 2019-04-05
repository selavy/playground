# src/CMakeLists.txt

add_executable(tests
    unit_test_main.cpp
    buckets.cpp
    )

target_link_libraries(tests PUBLIC
    Flags
    Catch2
    )
