cmake_minimum_required(VERSION 3.18)
project(kvpark_kvbridge LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libkvbridge: the translation engine's products (encode, section decode, RoPE, cache cast) as ggml
# graphs, behind a C API that backends/ggml_backend.py loads with ctypes. See README.md.
#
# Two ways to get ggml:
#   -DKVBRIDGE_GGML_SOURCE=.../llama.cpp/ggml   build ggml from source with this library (static,
#                                               its symbols hidden: one self-contained .so)
#   (unset)                                     find_package(ggml) of an installed ggml
set(KVBRIDGE_GGML_SOURCE "" CACHE PATH "a ggml source tree (llama.cpp/ggml) to build with the library")
option(KVBRIDGE_VULKAN "build ggml's Vulkan backend (with KVBRIDGE_GGML_SOURCE)" ON)
# off by default: on macOS the library's GPU is then Vulkan (MoltenVK), not Metal
option(KVBRIDGE_METAL "build ggml's Metal backend as well (macOS, with KVBRIDGE_GGML_SOURCE; untested)" OFF)

if(KVBRIDGE_GGML_SOURCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(GGML_VULKAN ${KVBRIDGE_VULKAN} CACHE BOOL "" FORCE)
    set(GGML_METAL ${KVBRIDGE_METAL} CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${KVBRIDGE_GGML_SOURCE}" ggml EXCLUDE_FROM_ALL)
    set(KVBRIDGE_GGML_LIBS ggml ggml-base ggml-cpu)
else()
    find_package(ggml REQUIRED)
    set(KVBRIDGE_GGML_LIBS ggml::ggml ggml::ggml-base ggml::ggml-cpu)
endif()

add_library(kvbridge SHARED kvbridge.cpp)
target_link_libraries(kvbridge PRIVATE ${KVBRIDGE_GGML_LIBS})
set_target_properties(kvbridge PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden
                                          VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(kvbridge PRIVATE KVBRIDGE_BUILD)
# only kvb_* is exported: a process that also loads another libggml keeps its own
if(KVBRIDGE_GGML_SOURCE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(kvbridge PRIVATE "-Wl,--exclude-libs,ALL")
elseif(KVBRIDGE_GGML_SOURCE AND APPLE)
    target_link_options(kvbridge PRIVATE "-Wl,-exported_symbol,_kvb_*")      # untested
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kvbridge PRIVATE -Wall -Wextra)
endif()
