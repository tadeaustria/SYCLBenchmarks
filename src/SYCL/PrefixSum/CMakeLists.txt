cmake_minimum_required(VERSION 3.0.0)
project(MAIN VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64-vulkan-windows-sycldevice")

add_executable(MAIN main.cpp)

include_directories(SYSTEM "$ENV{SYCL_BUILD}\\include\\sycl")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)