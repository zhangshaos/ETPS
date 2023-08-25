cmake_minimum_required(VERSION 3.12)

aux_source_directory(. SRC)
add_library(ETPS STATIC ${SRC})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ETPS
        PUBLIC
        OpenMP::OpenMP_CXX)
endif ()

target_include_directories(ETPS
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(ETPS
    PUBLIC
    opencv_core
    cpptrace)

target_compile_definitions(ETPS
    PUBLIC
    _USE_MATH_DEFINES)

set(ENABLE_CHECK_FLOAT_ERROR ON) #打开此项检查浮点数错误
if(NOT ${ENABLE_CHECK_FLOAT_ERROR})
    target_compile_definitions(ETPS
        PRIVATE
        N_SELF_CHECK_FLOAT)
endif()

if (NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()
set_target_properties(ETPS
    PROPERTIES
    DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})