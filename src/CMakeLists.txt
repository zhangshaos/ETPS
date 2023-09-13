cmake_minimum_required(VERSION 3.15)

#CCS library
aux_source_directory(. SRC)
add_library(CCS STATIC ${SRC})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CCS
        PUBLIC
        OpenMP::OpenMP_CXX)
endif ()

target_include_directories(CCS
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(CCS
    PUBLIC
    ${OpenCV_LIBS}
    cpptrace)

target_compile_definitions(CCS
    PUBLIC
    _USE_MATH_DEFINES)

set(ENABLE_CHECK_FLOAT_ERROR OFF) #打开此项检查浮点数错误
if(NOT ${ENABLE_CHECK_FLOAT_ERROR})
    target_compile_definitions(CCS
        PRIVATE
        N_SELF_CHECK_FLOAT)
endif()

set_target_properties(CCS
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

#Python wrapper for CCS
aux_source_directory(./pybind11 PY_SRC)
pybind11_add_module(pyCCS ${PY_SRC})
target_link_libraries(pyCCS PRIVATE CCS)