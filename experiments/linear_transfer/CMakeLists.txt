cmake_minimum_required(VERSION 3.16)
project(kvpark_collect_values LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_SOURCE "" CACHE PATH "Existing llama.cpp source directory")
set(LLAMA_LIBRARY_DIR "" CACHE PATH "Matching built llama.cpp library directory")
# the llama.cpp commit of LLAMA_SOURCE (build_collector.py passes it): the tools follow its loader rules
set(KVPARK_LLAMA_REVISION "" CACHE STRING "llama.cpp revision of LLAMA_SOURCE")
find_library(LLAMA_LIBRARY NAMES llama PATHS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIBRARY NAMES ggml PATHS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIBRARY NAMES ggml-base PATHS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)

function(kvpark_tool name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE
        "${LLAMA_SOURCE}/include" "${LLAMA_SOURCE}/ggml/include" "${LLAMA_SOURCE}/vendor")
    target_link_libraries(${name} PRIVATE "${LLAMA_LIBRARY}" "${GGML_LIBRARY}" "${GGML_BASE_LIBRARY}")
    target_compile_definitions(${name} PRIVATE KVPARK_LLAMA_REVISION="${KVPARK_LLAMA_REVISION}")
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${LLAMA_LIBRARY_DIR}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# v1 value collector (unchanged behavior)
kvpark_tool(kvpark-collect-values collector.cpp)
# v2 K/V probe: tokenize, collect, patch-score
kvpark_tool(kvpark-kv-probe kv_probe.cpp)
# state driver for any architecture: save / restore-decode / capture with token ids
kvpark_tool(kvpark-state-driver state_driver.cpp)
# llama.cpp's own vocabulary (vocab_only): pieces and tokenizations the engine's are checked against
kvpark_tool(kvpark-vocab vocab_tool.cpp)
# Optional: llama.cpp's own chat-template application (common_chat_templates_apply), as a reference
# for chat.py renders; needs a runtime built with LLAMA_BUILD_COMMON=ON (libllama-common).
find_library(LLAMA_COMMON_LIBRARY NAMES llama-common PATHS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH)
if(LLAMA_COMMON_LIBRARY)
    kvpark_tool(kvpark-chat-apply chat_apply.cpp)
    target_include_directories(kvpark-chat-apply PRIVATE "${LLAMA_SOURCE}/common")
    target_link_libraries(kvpark-chat-apply PRIVATE "${LLAMA_COMMON_LIBRARY}")
endif()
