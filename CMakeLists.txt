cmake_minimum_required(VERSION 3.15)
project(eyetracker_cplus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(eyetracker STATIC
    src/gaze_calibration.cpp
)

target_include_directories(eyetracker PUBLIC include)

add_executable(eyetracker_demo src/main.cpp)
target_link_libraries(eyetracker_demo PRIVATE eyetracker)

