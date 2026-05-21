if (NCCL_HAS_HOST_RMA)
    if(NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    endif()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 100)
    endif()
    enable_language(CUDA)
    file(GLOB XPORT_SOURCES "*.cpp")
    list(APPEND XPORT_SOURCES nccl_gin_device.cu)
    add_library(tent_xport_nccl STATIC ${XPORT_SOURCES})
    target_link_libraries(tent_xport_nccl PUBLIC tent_common platform_cuda ${NCCL_LIB})
    target_include_directories(tent_xport_nccl PUBLIC ${NCCL_INCLUDE_DIR})
    target_compile_definitions(tent_xport_nccl PRIVATE
        NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1)
    target_compile_options(tent_xport_nccl PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(tent_xport_nccl PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 100)
endif()
