import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
torch_lib_path = os.path.join(torch.__path__[0], "lib")
norm_ext = CUDAExtension(
    name="norm_ext",
    sources=[
        "norm_ext.cu",
    ],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
        ],
    },
    extra_link_args=[f"-Wl,-rpath,{torch_lib_path}"],
)

setup(
    name="byoxvllm_norm_ext",
    ext_modules=[norm_ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
