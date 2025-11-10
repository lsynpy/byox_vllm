#include <cuda_fp16.h>
#include <torch/extension.h>

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T, size_t vec_size>
struct vec_t;

template <size_t vec_size>
struct vec_t<half, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  int4 data[vec_size / 8];
  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)data)[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)data)[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(half2*)(&(data[i].x)) = make_half2(val, val);
      *(half2*)(&(data[i].y)) = make_half2(val, val);
      *(half2*)(&(data[i].z)) = make_half2(val, val);
      *(half2*)(&(data[i].w)) = make_half2(val, val);
    }
  }
  FLASHINFER_INLINE void load(const half* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((int4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(half* ptr) const {
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
      printf("%f", __half2float((*this)[i]));
    }
    printf("]\n");
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
__global__ void RMSNormKernelHalf(const __half* __restrict__ input,
                                  const __half* __restrict__ weight, __half* __restrict__ output,
                                  uint32_t d, uint32_t stride_input, uint32_t stride_output,
                                  float eps) {
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
    vec_t<half, VEC_SIZE> input_vec;
    input_vec.fill(__float2half(0.f));
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
    // if (thread_id == 0 && blockIdx.x == 0) {
    //   input_vec.print("input_vec = ");
    // }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += __half2float(input_vec[j]) * __half2float(input_vec[j]);
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
    vec_t<half, VEC_SIZE> input_vec;
    vec_t<half, VEC_SIZE> weight_vec;
    vec_t<half, VEC_SIZE> output_vec;
    input_vec.fill(__float2half(0.f));
    weight_vec.fill(__float2half(0.f));
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] =
          __float2half(__half2float(input_vec[j]) * rms_rcp * __half2float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
}

void rmsnorm_fp16(torch::Tensor output, torch::Tensor input, torch::Tensor weight, double eps) {
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);

  cudaSetDevice(input.get_device());

  constexpr uint32_t vec_size = 128 / sizeof(half) / 8;  // 8 = LDG.E.128 / 2 / 8
  const uint32_t block_size = std::min<uint32_t>(1024, hidden_size / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  // printf("vec_size: %d, block_size: %d, num_warps %d, smem_size: %d\n", vec_size, block_size,
  //  num_warps, smem_size);

  RMSNormKernelHalf<vec_size>
      <<<batch_size, dim3(32, num_warps)>>>(reinterpret_cast<const __half*>(input.data_ptr()),
                                            reinterpret_cast<const __half*>(weight.data_ptr()),
                                            reinterpret_cast<__half*>(output.data_ptr()),
                                            hidden_size,       //
                                            input.stride(0),   //
                                            output.stride(0),  //
                                            static_cast<float>(eps));

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "RMSNorm kernel launch failed: ", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rmsnorm", &rmsnorm_fp16, "RMSNorm FP16"); }
