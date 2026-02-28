cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(Sam3CPP)
find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

add_library(sam3_cpp_lib SHARED sam3.h sam3.cpp util.h util.cpp)
if (APPLE)
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib)
else()
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
endif()
set(libsentencepiece_lib ${TOKENIZERS_ROOT_DIR}/libs_universal/libsentencepiece.a)
set(libtokenizers_c_lib ${TOKENIZERS_ROOT_DIR}/libs_universal/libtokenizers_c.a)
set(libtokenizers_cpp_lib ${TOKENIZERS_ROOT_DIR}/libs_universal/libtokenizers_cpp.a)

target_include_directories(
  sam3_cpp_lib PUBLIC 
  ${ONNXRUNTIME_ROOT_DIR}/include
  ${TOKENIZERS_ROOT_DIR}/include
)

target_link_libraries(
  sam3_cpp_lib PUBLIC
  ${onnxruntime_lib}
  ${libsentencepiece_lib}
  ${libtokenizers_c_lib}
  ${libtokenizers_cpp_lib}
  ${OpenCV_LIBS}
  gflags
)

add_executable(sam3_cpp_test test.cpp)
target_link_libraries(
  sam3_cpp_test PRIVATE
  sam3_cpp_lib
)

