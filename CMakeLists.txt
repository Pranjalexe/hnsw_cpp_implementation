set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(include)

add_library(hnsw_lib STATIC src/hnsw.cpp)
target_compile_options(hnsw_lib PRIVATE -O3 -march=native -funroll-loops -ffast-math -Wall -Wextra)

add_executable(test_hnsw src/test.cpp)
target_link_libraries(test_hnsw PRIVATE hnsw_lib)
target_compile_options(test_hnsw PRIVATE -O2 -march=native -Wall -Wextra)

add_executable(bench src/main.cpp)
target_link_libraries(bench PRIVATE hnsw_lib)
target_compile_options(bench PRIVATE -O3 -march=native -funroll-loops -ffast-math)

enable_testing()
add_test(NAME correctness COMMAND test_hnsw)
