cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Vulkan 1.2.141 REQUIRED)
find_package(OpenMP)

message("Version ${Vulkan_VERSION} ${Vulkan_VERSION_MAJOR}")

add_executable(main main.cpp)

if(OpenMP_CXX_FOUND)
  message(STATUS "OMP found")
  target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX Vulkan::Vulkan)
else()
  message(STATUS "OMP not found")
  target_link_libraries(main Vulkan::Vulkan)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)