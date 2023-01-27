cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
project(poco_cv)
set(CMAKE_CXX_STANDARD 14)

add_executable(poco_cv src/main.cpp src/onnx_runner.h src/onnx_runner.cpp)

set(POCO_PREFIX "/poco/cmake-build/install")
set(POCO_INCLUDE_DIR "${POCO_PREFIX}/include")
set(POCO_LIB_DIR "${POCO_PREFIX}/lib")

target_include_directories(poco_cv PRIVATE "${POCO_INCLUDE_DIR}")
target_link_libraries(poco_cv "${POCO_LIB_DIR}/libPocoFoundation.so")
target_link_libraries(poco_cv "${POCO_LIB_DIR}/libPocoNet.so")
target_link_libraries(poco_cv "${POCO_LIB_DIR}/libPocoUtil.so")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( poco_cv ${OpenCV_LIBS} )

set(ONNX_RUNTIME_PREFIX "/onnxruntime-linux-x64-1.13.1/")
target_include_directories(poco_cv PRIVATE "${ONNX_RUNTIME_PREFIX}/include")
target_link_libraries(poco_cv "${ONNX_RUNTIME_PREFIX}/lib/libonnxruntime.so")
target_link_libraries(poco_cv "${ONNX_RUNTIME_PREFIX}/lib/libonnxruntime.so.1.13.1")
