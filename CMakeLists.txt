cmake_minimum_required(VERSION 3.0)
project(dstag)

FIND_PACKAGE( Boost COMPONENTS program_options regex serialization filesystem REQUIRED )

# TODO: change this, it overrides instead of adding flags !
add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-pedantic")
#add_definitions("-ftemplate-depth=512")
#add_definitions("-O3")
add_definitions("-Wfatal-errors")
add_definitions("-g")
add_definitions("-std=c++11")
#add_definitions("-funroll-loops")
add_definitions("-DEIGEN_FAST_MATH")
#add_definitions("-march=native")


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
include_directories("${PROJECT_SOURCE_DIR}/src")
INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIR}")

add_library(graph ${PROJECT_SOURCE_DIR}/src/graph.cpp)
set_property(TARGET graph PROPERTY CXX_STANDARD 11)

add_library(dependency ${PROJECT_SOURCE_DIR}/src/dependency.cpp)
set_property(TARGET dependency PROPERTY CXX_STANDARD 11)


add_executable(dep-train-tagger ${PROJECT_SOURCE_DIR}/src/dep_train_tagger.cpp)
set_property(TARGET dep-train-tagger PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(dep-train-tagger ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(dep-train-tagger dynet)
TARGET_LINK_LIBRARIES(dep-train-tagger graph)
TARGET_LINK_LIBRARIES(dep-train-tagger dependency)

add_executable(dep-train-parser ${PROJECT_SOURCE_DIR}/src/dep_train_parser.cpp)
set_property(TARGET dep-train-parser PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(dep-train-parser ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(dep-train-parser dynet)
TARGET_LINK_LIBRARIES(dep-train-parser graph)
TARGET_LINK_LIBRARIES(dep-train-parser dependency)

add_executable(dep-build-dict ${PROJECT_SOURCE_DIR}/src/dep_build_dict.cpp)
set_property(TARGET dep-build-dict PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(dep-build-dict ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(dep-build-dict dynet)

add_executable(dep-build-filter ${PROJECT_SOURCE_DIR}/src/dep_build_filter.cpp)
set_property(TARGET dep-build-filter PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(dep-build-filter ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(dep-build-filter dynet)

add_executable(dep-compute-attachment-probs ${PROJECT_SOURCE_DIR}/src/dep_compute_attachment_probs.cpp)
set_property(TARGET dep-compute-attachment-probs PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(dep-compute-attachment-probs ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(dep-compute-attachment-probs dynet)

add_executable(dep-decode-pipeline ${PROJECT_SOURCE_DIR}/src/dep_decode_pipeline.cpp)
set_property(TARGET dep-decode-pipeline PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(dep-decode-pipeline ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(dep-decode-pipeline dynet)
TARGET_LINK_LIBRARIES(dep-decode-pipeline graph)
TARGET_LINK_LIBRARIES(dep-decode-pipeline dependency)

add_executable(dep-decode-joint ${PROJECT_SOURCE_DIR}/src/dep_decode_joint.cpp)
set_property(TARGET dep-decode-joint PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(dep-decode-joint ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(dep-decode-joint dynet)
TARGET_LINK_LIBRARIES(dep-decode-joint graph)
TARGET_LINK_LIBRARIES(dep-decode-joint dependency)




add_executable(spine-build-dict ${PROJECT_SOURCE_DIR}/src/spine_build_dict.cpp)
set_property(TARGET spine-build-dict PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(spine-build-dict ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(spine-build-dict dynet)

add_executable(spine-train-tagger ${PROJECT_SOURCE_DIR}/src/spine_train_tagger.cpp)
set_property(TARGET spine-train-tagger PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(spine-train-tagger ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(spine-train-tagger dynet)
TARGET_LINK_LIBRARIES(spine-train-tagger graph)
TARGET_LINK_LIBRARIES(spine-train-tagger dependency)


add_executable(spine-train-tagger-head ${PROJECT_SOURCE_DIR}/src/spine_train_tagger_head.cpp)
set_property(TARGET spine-train-tagger-head PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(spine-train-tagger-head ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(spine-train-tagger-head dynet)
TARGET_LINK_LIBRARIES(spine-train-tagger-head graph)
TARGET_LINK_LIBRARIES(spine-train-tagger-head dependency)

add_executable(spine-train-parser ${PROJECT_SOURCE_DIR}/src/spine_train_parser.cpp)
set_property(TARGET spine-train-parser PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(spine-train-parser ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(spine-train-parser dynet)
TARGET_LINK_LIBRARIES(spine-train-parser graph)
TARGET_LINK_LIBRARIES(spine-train-parser dependency)

add_executable(spine-compute-attachment-probs ${PROJECT_SOURCE_DIR}/src/spine_compute_attachment_probs.cpp)
set_property(TARGET spine-compute-attachment-probs PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(spine-compute-attachment-probs ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(spine-compute-attachment-probs dynet)


add_executable(spine-decode-joint ${PROJECT_SOURCE_DIR}/src/spine_decode_joint.cpp)
set_property(TARGET spine-decode-joint PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(spine-decode-joint ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(spine-decode-joint dynet)
TARGET_LINK_LIBRARIES(spine-decode-joint graph)
TARGET_LINK_LIBRARIES(spine-decode-joint dependency)

add_executable(spine-build-filter ${PROJECT_SOURCE_DIR}/src/spine_build_filter.cpp)
set_property(TARGET spine-build-filter PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(spine-build-filter ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(spine-build-filter dynet)

add_executable(spine-decode-pipeline ${PROJECT_SOURCE_DIR}/src/spine_decode_pipeline.cpp)
set_property(TARGET spine-decode-pipeline PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(spine-decode-pipeline ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(spine-decode-pipeline dynet)
TARGET_LINK_LIBRARIES(spine-decode-pipeline graph)
TARGET_LINK_LIBRARIES(spine-decode-pipeline dependency)

