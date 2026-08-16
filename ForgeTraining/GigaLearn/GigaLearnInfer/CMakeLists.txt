cmake_minimum_required (VERSION 3.8)

project("GigaLearnInfer")

# Add all headers and code files

set(FILES_SRC
    src/private/Agent.cpp
    src/private/AgentSetupSnapshot.cpp
    src/private/InferUnit.cpp
    src/private/ModelManager.cpp
    src/private/Models.cpp
)

add_library(GigaLearnInfer STATIC ${FILES_SRC})
target_compile_definitions(GigaLearnInfer PUBLIC -DGIGALEARN_INFER_STATIC)
target_compile_definitions(GigaLearnInfer PRIVATE -DWITHIN_GGL)
target_compile_definitions(GigaLearnInfer PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
target_include_directories(GigaLearnInfer PUBLIC
    "src/public/"
    "../libsrc/json")

# Allow override: -DLIBTORCH_ROOT="C:/path/to/libtorch"
if(NOT DEFINED LIBTORCH_ROOT)
    if(WIN32)
        set(
            LIBTORCH_ROOT
            "$ENV{LOCALAPPDATA}/libtorch"
            CACHE PATH
            "Root directory of the libtorch distribution")
    else()
        set(
            LIBTORCH_ROOT
            "/opt/libtorch"
            CACHE PATH
            "Root directory of the libtorch distribution")
    endif()
endif() 

# TorchConfig.cmake usually lives here in the official libtorch distribution:
set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch")

if(NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
message(FATAL_ERROR
    "Could not find TorchConfig.cmake at: ${Torch_DIR}/TorchConfig.cmake\n"
    "Check LIBTORCH_ROOT='${LIBTORCH_ROOT}' (expected libtorch layout with share/cmake/Torch)."
)
endif()

find_package(Torch REQUIRED CONFIG)
 
if(TARGET Torch::Torch)
  set(_TORCH_LINK Torch::Torch)
elseif(TARGET torch)
  set(_TORCH_LINK torch)
else()
  set(_TORCH_LINK ${TORCH_LIBRARIES})
endif() 


if (TORCH_CUDA_LIBRARIES)
	message("Enabling CUDA support...")
	target_compile_definitions(GigaLearnInfer PUBLIC -DRG_CUDA_SUPPORT)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(GigaLearnInfer PUBLIC ${_TORCH_LINK})

if(WIN32)
  if(TORCH_LIBRARY)
    get_filename_component(_TORCH_RUNTIME_LIBRARY_DIR "${TORCH_LIBRARY}" DIRECTORY)
  else()
    set(_TORCH_RUNTIME_LIBRARY_DIR "${LIBTORCH_ROOT}/lib")
  endif()

  file(GLOB _TORCH_RUNTIME_DLLS "${_TORCH_RUNTIME_LIBRARY_DIR}/*.dll")

  if(_TORCH_RUNTIME_DLLS)
    set(GIGALEARN_INFER_TORCH_RUNTIME_DIR "${_TORCH_RUNTIME_LIBRARY_DIR}" PARENT_SCOPE)
    set(GIGALEARN_INFER_TORCH_RUNTIME_DLLS ${_TORCH_RUNTIME_DLLS} PARENT_SCOPE)
  else()
    message(WARNING "No Torch runtime DLLs found in ${_TORCH_RUNTIME_LIBRARY_DIR}")
  endif()
endif()

# Set C++ version to 20
set_target_properties(GigaLearnInfer PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnInfer PROPERTIES CXX_STANDARD 20)

target_link_libraries(GigaLearnInfer PUBLIC RLGym)
