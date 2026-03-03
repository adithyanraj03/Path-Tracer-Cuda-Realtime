cmake_minimum_required(VERSION 3.21)
include(FetchContent)

# Point CMake to CUDA 13.1 (compatible with MSVC 19.43)
set(CMAKE_CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe")

project(cuda_path_tracer VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86")

# Directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gl/include)

# Files
file(GLOB SOURCES "src/*.cpp")
file(GLOB CUDA_SOURCES "src/*.cu")
set(GLAD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/gl/src/glad.c")

# Fetch GLFW automatically
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Executable
add_executable(cuda_path_tracer ${SOURCES} ${GLAD_SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(cuda_path_tracer PRIVATE glfw ${CMAKE_DL_LIBS})

# Windows needs opengl32 explicitly
if(WIN32)
    target_link_libraries(cuda_path_tracer PRIVATE opengl32)
endif()

# Scope MSVC warning suppression to C++ files only (not nvcc)
if(MSVC)
    set_source_files_properties(${SOURCES} PROPERTIES COMPILE_FLAGS "/wd4244 /wd4267")
endif()
