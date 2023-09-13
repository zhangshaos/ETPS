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


add_executable(t_lsd t.cpp)
target_link_libraries(t_lsd
    PRIVATE
    ${OpenCV_LIBS}
)


add_executable(demo_sem_seg sem_seg_main.cpp)
target_link_libraries(demo_sem_seg
    PRIVATE
    CCS
    ${OpenCV_LIBS}
)