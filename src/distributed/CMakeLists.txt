# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g")
set(STA_DEPS data base pslite protobuf zmq pthread)
add_library(distributed STATIC dist_score_function.cc dist_linear_score.cc dist_fm_score.cc dist_ffm_score.cc dist_cross_entropy_loss.cc dist_checker.cc dist_loss.cc dist_score_test.cc)
target_link_libraries(distributed ${STA_DEPS})

add_executable(dist_xlearn_train dist_train_main.cc)
target_link_libraries(dist_xlearn_train solver distributed)

install(TARGETS distributed DESTINATION lib/distributed)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
message(${HEADER_FILES})
install(FILES ${HEADER_FILES} DESTINATION include/distributed)

# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/dist)

# Build uinttests
set(LIBS distributed gtest)

add_executable(dist_score_function_test dist_score_test.cc)
target_link_libraries(dist_score_function_test gtest_main ${LIBS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/distributed)

# Build static library
set(STA_DEPS base)
add_library(distributed STATIC parameter_server.cc)
target_link_libraries(distributed ${STA_DEPS})

# Build unittests.
set(LIBS distributed data base gtest)

add_executable(parameter_server_test parameter_server_test.cc)
target_link_libraries(parameter_server_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS distributed DESTINATION lib/distributed)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/distributed)
