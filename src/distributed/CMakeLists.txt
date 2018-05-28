# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/distributed)

# Build static library
set(STA_DEPS base pslite protobuf zmq pthread)
add_library(distributed STATIC parameter_server.cc)
target_link_libraries(distributed ${STA_DEPS})

# Build unittests.
set(LIBS distributed data base gtest pslite protobuf zmq pthread)

add_executable(parameter_server_test parameter_server_test.cc)
target_link_libraries(parameter_server_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS distributed DESTINATION lib/distributed)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/distributed)

# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(STA_DEPS data base pslite protobuf zmq pthread)

add_executable(dist_xlearn_train dist_train_main.cc server_handler.h)
target_link_libraries(dist_xlearn_train solver distributed)
