cmake_minimum_required(VERSION 3.12)
project(CCS-Demo)

find_package(argparse CONFIG REQUIRED)

add_executable(demo main.cpp)

target_link_libraries(demo
    PRIVATE
    CCS
    ${OpenCV_LIBS}
    argparse::argparse
)

if (NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()
set_target_properties(demo
    PROPERTIES
    DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
)