cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_model)

find_package(Torch REQUIRED)

add_executable(test_model test_model.cpp)
target_link_libraries(test_model "${TORCH_LIBRARIES}")
set_property(TARGET test_model PROPERTY CXX_STANDARD 14)
