cmake_minimum_required(VERSION 3.16)

option(WITH_PG_CORE_PROTOTYPE "Build the Torch-free Mooncake P2P core probe" OFF)

if(WITH_PG_CORE_PROTOTYPE)
  enable_language(CUDA)
  if(NOT TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
  endif()
  if(NOT TORCH_CUDA_ARCH_LIST)
    message(FATAL_ERROR "pg_core requires TORCH_CUDA_ARCH_LIST (for example: 8.0;9.0)")
  endif()
  string(REPLACE "+PTX" "" _pg_core_cuda_architectures "${TORCH_CUDA_ARCH_LIST}")
  string(REPLACE "." "" _pg_core_cuda_architectures "${_pg_core_cuda_architectures}")
  pybind11_add_module(pg_core
    ../src/p2p_core_py.cpp
    ../src/p2p_core.cpp
    ../src/core_cuda_collective_worker.cpp
    ../src/collective_reduce.cpp
    ../src/collective_reduce_cuda.cu
    ../src/core_cuda_task_kernel.cu
    ../src/p2p_proxy.cpp
    ../src/connection_poller.cpp)
  target_include_directories(pg_core PRIVATE
    ../include
    ../../mooncake-transfer-engine/include)
  # The legacy PG and EP both consume engine.so. Linking a second static copy
  # of TransferEngine into pg_core creates an independent RDMA runtime inside
  # the process, which is not a valid production resource topology.
  add_dependencies(pg_core engine)
  set(_pg_core_staging_dir "${CMAKE_BINARY_DIR}/pg_core_staging")
  add_custom_command(TARGET pg_core PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_pg_core_staging_dir}"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${_pg_core_staging_dir}/engine.so"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "$<TARGET_FILE:engine>" "${_pg_core_staging_dir}/engine.so")
  target_link_options(pg_core PRIVATE
    "-L${_pg_core_staging_dir}"
    "-Wl,--push-state,--no-as-needed"
    "-l:engine.so"
    "-Wl,--pop-state")
  target_link_libraries(pg_core PRIVATE ibverbs mlx5)
  set_property(TARGET pg_core PROPERTY CUDA_ARCHITECTURES "${_pg_core_cuda_architectures}")
  set_target_properties(pg_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_pg_core_staging_dir}"
    BUILD_RPATH "$ORIGIN")
endif()
