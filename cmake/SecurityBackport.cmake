# Preserve the Windows CUDA stack when building the maintained source package.
if(WIN32 AND "${TORCH_BUILD_VERSION}" STREQUAL "2.3.0+security.1")
  if(NOT MSVC OR NOT USE_CUDA OR NOT CAFFE2_USE_CUDA OR NOT CAFFE2_USE_CUDNN
      OR NOT CUDA_VERSION VERSION_EQUAL "12.1"
      OR NOT CUDNN_VERSION VERSION_EQUAL "8.9.2")
    message(FATAL_ERROR
      "The Windows security backport requires MSVC, CUDA 12.1 and cuDNN 8.9.2. "
      "Install the matching toolchain before building; CPU substitution is unavailable.")
  endif()
endif()
