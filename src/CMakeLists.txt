cmake_minimum_required(VERSION 3.12)

aux_source_directory(. SRC)
add_library(FH STATIC ${SRC})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(FH
        PUBLIC
        OpenMP::OpenMP_CXX)
endif ()

target_include_directories(FH
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(FH
    PUBLIC
    opencv_core
    cpptrace)

target_compile_definitions(FH
    PUBLIC
    _USE_MATH_DEFINES)

set(ENABLE_CHECK_FLOAT_ERROR ON) #打开此项检查浮点数错误
if(NOT ${ENABLE_CHECK_FLOAT_ERROR})
    target_compile_definitions(FH
        PRIVATE
        N_SELF_CHECK_FLOAT)
endif()

if (NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()
set_target_properties(FH
    PROPERTIES
    DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})