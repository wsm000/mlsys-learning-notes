#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    const hipError_t status = (call);                                          \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(status) << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (false)

__global__ void vector_add(const float* a, const float* b, float* c, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) *
                       static_cast<size_t>(blockDim.x) +
                   static_cast<size_t>(threadIdx.x);
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

size_t parse_positive_size(const char* text, const char* argument_name) {
  try {
    const std::string value{text};
    size_t parsed_characters = 0;
    const unsigned long long parsed = std::stoull(value, &parsed_characters);
    if (parsed_characters != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("not a positive size");
    }
    return static_cast<size_t>(parsed);
  } catch (const std::exception&) {
    std::cerr << "Invalid " << argument_name << ": " << text
              << ". Expected a positive integer." << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  const size_t n = argc >= 2 ? parse_positive_size(argv[1], "N") : 10000000;
  const size_t block_size =
      argc >= 3 ? parse_positive_size(argv[2], "block size") : 256;

  if (argc > 3) {
    std::cerr << "Usage: " << argv[0] << " [N] [block_size]" << std::endl;
    return EXIT_FAILURE;
  }

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, device));

  if (block_size > static_cast<size_t>(properties.maxThreadsPerBlock)) {
    std::cerr << "Block size " << block_size << " exceeds device limit "
              << properties.maxThreadsPerBlock << std::endl;
    return EXIT_FAILURE;
  }

  const size_t grid_size = n / block_size + (n % block_size != 0);
  if (grid_size > std::numeric_limits<unsigned int>::max()) {
    std::cerr << "Grid size is too large for a one-dimensional HIP launch."
              << std::endl;
    return EXIT_FAILURE;
  }
  if (n > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "N is too large for host/device allocation." << std::endl;
    return EXIT_FAILURE;
  }

  const size_t bytes = n * sizeof(float);
  std::vector<float> host_a(n);
  std::vector<float> host_b(n);
  std::vector<float> host_c(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    host_a[i] = static_cast<float>(i % 97);
    host_b[i] = static_cast<float>((i * 3) % 89);
  }

  std::cout << "=== HIP Vector Add ===" << std::endl;
  std::cout << "gpu_device: " << device << std::endl;
  std::cout << "gpu_name: " << properties.name << std::endl;
  std::cout << "gpu_arch: " << properties.gcnArchName << std::endl;
  std::cout << "compute_units: " << properties.multiProcessorCount << std::endl;
  std::cout << "N: " << n << std::endl;
  std::cout << "block_size: " << block_size << std::endl;
  std::cout << "grid_size: " << grid_size << std::endl;
  std::cout << "index_formula: blockIdx.x * blockDim.x + threadIdx.x"
            << std::endl;

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_a), bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_b), bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_c), bytes));

  HIP_CHECK(hipMemcpy(device_a, host_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_b, host_b.data(), bytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(vector_add, dim3(static_cast<unsigned int>(grid_size)),
                     dim3(static_cast<unsigned int>(block_size)), 0, 0,
                     device_a, device_b, device_c, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(host_c.data(), device_c, bytes, hipMemcpyDeviceToHost));

  size_t error_count = 0;
  size_t first_error = 0;
  for (size_t i = 0; i < n; ++i) {
    const float expected = host_a[i] + host_b[i];
    if (std::fabs(host_c[i] - expected) > 1.0e-6f) {
      if (error_count == 0) {
        first_error = i;
      }
      ++error_count;
    }
  }

  HIP_CHECK(hipFree(device_c));
  HIP_CHECK(hipFree(device_b));
  HIP_CHECK(hipFree(device_a));

  if (error_count != 0) {
    std::cerr << "RESULT: FAIL (errors=" << error_count
              << ", first_index=" << first_error << ", expected="
              << host_a[first_error] + host_b[first_error] << ", actual="
              << host_c[first_error] << ")" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "RESULT: PASS (N=" << n << ", block=" << block_size
            << ", grid=" << grid_size << ")" << std::endl;
  return EXIT_SUCCESS;
}
