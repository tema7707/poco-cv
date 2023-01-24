cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

project(web_server)

set(CMAKE_CXX_STANDARD 14)

add_executable(src/main.cpp)

set(POCO_PREFIX "/poco/cmake-build/install")
#set(POCO_PREFIX "/Users/artemiy/Projects/My/poco/cmake-build/install")
set(POCO_INCLUDE_DIR "${POCO_PREFIX}/include")
set(POCO_LIB_DIR "${POCO_PREFIX}/lib")

target_include_directories(web_server PRIVATE "${POCO_INCLUDE_DIR}")
target_link_libraries(web_server "${POCO_LIB_DIR}/libPocoFoundation.so")
target_link_libraries(web_server "${POCO_LIB_DIR}/libPocoNet.so")
target_link_libraries(web_server "${POCO_LIB_DIR}/libPocoUtil.so")

#target_link_libraries(web_server "${POCO_LIB_DIR}/libPocoFoundation.dylib")
#target_link_libraries(web_server "${POCO_LIB_DIR}/libPocoNet.dylib")
#target_link_libraries(web_server "${POCO_LIB_DIR}/libPocoUtil.dylib")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( web_server ${OpenCV_LIBS} )

#set(ONNX_RUNTIME_PREFIX "/Users/artemiy/Downloads/onnxruntime-osx-universal2-1.13.1/")
set(ONNX_RUNTIME_PREFIX "/onnxruntime-linux-x64-1.13.1/")
target_include_directories(web_server PRIVATE "${ONNX_RUNTIME_PREFIX}/include")
#target_link_libraries(web_server "${ONNX_RUNTIME_PREFIX}/lib/libonnxruntime.dylib")
target_link_libraries(web_server "${ONNX_RUNTIME_PREFIX}/lib/libonnxruntime.so")
#target_link_libraries(web_server "${ONNX_RUNTIME_PREFIX}/lib/libonnxruntime.1.13.1.dylib")
target_link_libraries(web_server "${ONNX_RUNTIME_PREFIX}/lib/libonnxruntime.so.1.13.1")
