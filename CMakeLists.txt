cmake_minimum_required(VERSION 3.15)
project(CCS VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
if(MSVC)
    add_compile_options(/utf-8 /bigobj /Zi)
else()
    add_compile_options(-g)
endif()

find_package(OpenCV CONFIG REQUIRED)

add_subdirectory(third_party/cpptrace)
set(PYBIND11_FINDPYTHON ON) #find_package(Python ...), set Python_ROOT_DIR firstly!
add_subdirectory(third_party/pybind11)
add_subdirectory(src)
add_subdirectory(demo)