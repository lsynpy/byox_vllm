#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T, size_t vec_size>
struct vec_t;

template <size_t vec_size>
struct vec_t<nv_bfloat16, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  int4 data[vec_size / 8];
  FLASHINFER_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)data)[i]; }
  FLASHINFER_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)data)[i];
  }
  FLASHINFER_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(nv_bfloat16 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(nv_bfloat162*)(&(data[i].x)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].y)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].z)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].w)) = make_bfloat162(val, val);
    }
  }
  FLASHINFER_INLINE void load(const nv_bfloat16* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((int4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(nv_bfloat16* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((int4*)ptr)[i] = data[i];
    }
  }
  FLASHINFER_INLINE void print(const char* prefix = "") const {
    printf("%s[", prefix);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      if (i > 0) printf(", ");
      printf("%f", __bfloat162float((*this)[i]));
    }
    printf("]\n");
  }
};

template <size_t vec_size>
struct vec_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  float4 data[vec_size / 4];
  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  FLASHINFER_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
};

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

template <uint32_t VEC_SIZE>
__global__ void RMSNormKernelBf16(const nv_bfloat16* __restrict__ input,
                                  const nv_bfloat16* __restrict__ weight,
                                  nv_bfloat16* __restrict__ output, uint32_t d,
                                  uint32_t stride_input, uint32_t stride_output, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;  // DONE: Will hurt performance if change to blockDim.x?
  // No, ptx & sass code show the loop is unrolled, but over/full unroll, see md docs.
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * blockDim.x;
  const uint32_t num_threads = blockDim.y * blockDim.x;
  // DONE: explain rounds, see md docs
  const uint32_t rounds = (d + VEC_SIZE * num_threads - 1) / (VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

  // start get element-wise square
  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<nv_bfloat16, VEC_SIZE> input_vec;
    input_vec.fill(__float2bfloat16(0.f));
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
    // if (thread_id == 0 && blockIdx.x == 0) {
    //   input_vec.print("input_vec = ");
    // }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += __bfloat162float(input_vec[j]) * __bfloat162float(input_vec[j]);
    }
  }
  // printf("sum_sq: %f\n", sum_sq);
  // end get element-wise square

// warp reduce
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();

  // inter-warp reduce
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrtf(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<nv_bfloat16, VEC_SIZE> input_vec;
    vec_t<nv_bfloat16, VEC_SIZE> weight_vec;
    vec_t<nv_bfloat16, VEC_SIZE> output_vec;
    input_vec.fill(__float2bfloat16(0.f));
    weight_vec.fill(__float2bfloat16(0.f));
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = __float2bfloat16(__bfloat162float(input_vec[j]) * rms_rcp *
                                       __bfloat162float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
}

void rmsnorm_bf16(torch::Tensor output, torch::Tensor input, torch::Tensor weight, float eps) {
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);

  cudaSetDevice(input.get_device());

  constexpr uint32_t vec_size = 128 / sizeof(nv_bfloat16) / 8;  // 8 = LDG.E.128 / 2 / 8
  const uint32_t block_size = std::min<uint32_t>(1024, hidden_size / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  const uint32_t smem_size = num_warps * sizeof(float);
  // printf("vec_size: %d, block_size: %d, num_warps %d, smem_size: %d\n", vec_size, block_size,
  //  num_warps, smem_size);

  cudaFuncSetAttribute(RMSNormKernelBf16<vec_size>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);
  RMSNormKernelBf16<vec_size><<<batch_size, dim3(32, num_warps), smem_size>>>(
      reinterpret_cast<const nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(output.data_ptr()), hidden_size, input.stride(0),
      output.stride(0), static_cast<float>(eps));

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "RMSNorm kernel launch failed: ", cudaGetErrorString(err));
  }
}

template <uint32_t VEC_SIZE>
__global__ void FusedAddRMSNormKernelBf16(nv_bfloat16* __restrict__ input,
                                          nv_bfloat16* __restrict__ residual,
                                          const nv_bfloat16* __restrict__ weight, const uint32_t d,
                                          const uint32_t stride_input,
                                          const uint32_t stride_residual, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<nv_bfloat16, VEC_SIZE> input_vec;
    input_vec.fill(__float2bfloat16(0.f));
    vec_t<nv_bfloat16, VEC_SIZE> residual_vec;
    residual_vec.fill(__float2bfloat16(0.f));
    vec_t<float, VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = __bfloat162float(input_vec[j]);
      x += __bfloat162float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = __float2bfloat16(x);
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      // FIXED: warp out-of-range address.
      // Need to call `cudaFuncSetAttribute`, and set smem_size in <<<...>>> to set smem more than
      // 48k https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrtf(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<nv_bfloat16, VEC_SIZE> input_vec;
    vec_t<nv_bfloat16, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> x_vec;
    input_vec.fill(__float2bfloat16(0.f));
    weight_vec.fill(__float2bfloat16(0.f));
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = __float2bfloat16(x_vec[j] * rms_rcp * __bfloat162float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
    }
  }
}

void fused_add_rmsnorm_bf16(torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
                            float eps = 1e-5) {
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);

  cudaSetDevice(input.get_device());

  constexpr uint32_t vec_size = 128 / sizeof(nv_bfloat16) / 8;  // 8 = LDG.E.128 / 2 / 8
  const uint32_t block_size = std::min<uint32_t>(1024, hidden_size / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  const uint32_t smem_size = (ceil_div(num_warps, 4) * 4 + hidden_size) * sizeof(float);
  // printf("vec_size: %d, block_size: %d, num_warps: %d, smem_size: %d floats + %d floats\n",
  //        vec_size, block_size, num_warps, ceil_div(num_warps, 4) * 4, hidden_size);

  cudaFuncSetAttribute(FusedAddRMSNormKernelBf16<vec_size>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  FusedAddRMSNormKernelBf16<vec_size><<<batch_size, dim3(32, num_warps), smem_size>>>(
      reinterpret_cast<nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(residual.data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(weight.data_ptr()), hidden_size, input.stride(0),
      residual.stride(0), static_cast<float>(eps));

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "RMSNorm kernel launch failed: ", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm_bf16, "RMSNorm BF16");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm_bf16, "Fused Add RMSNorm BF16");
}
