cmake_minimum_required(VERSION 3.20)
project(PrintVision LANGUAGES CXX)
find_package(OpenCV 4.5.4 EXACT REQUIRED COMPONENTS core imgproc ml)
add_library(PrintVision SHARED print.cpp refine.cpp)
target_compile_features(PrintVision PRIVATE cxx_std_17)
target_link_libraries(PrintVision PRIVATE ${OpenCV_LIBS})
target_compile_options(PrintVision PRIVATE /utf-8 /W4)

target_compile_definitions(PrintVision PRIVATE NOMINMAX)
