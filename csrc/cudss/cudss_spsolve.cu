/**
 * cuDSS (NVIDIA Direct Sparse Solver) backend for torch-sla
 * 
 * cuDSS is NVIDIA's high-performance direct sparse solver library that provides
 * GPU-accelerated LU, Cholesky, and LDLT factorizations.
 * 
 * NOTE: cuDSS requires the NVIDIA cuDSS library to be installed separately.
 * pip install nvidia-cudss-cu12  (for CUDA 12.x)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>

// Try to include cuDSS header - this may fail if cuDSS is not installed
#if __has_include(<cudss.h>)
#include <cudss.h>
#define CUDSS_AVAILABLE 1
#else
#define CUDSS_AVAILABLE 0
#endif

// Error checking macros
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

#define CHECK_CUSPARSE(call)                                                   \
    do {                                                                       \
        cusparseStatus_t status = call;                                        \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            throw std::runtime_error(std::string("cuSPARSE error: ") +         \
                                     std::to_string(status));                  \
        }                                                                      \
    } while (0)

#if CUDSS_AVAILABLE
#define CHECK_CUDSS(call)                                                      \
    do {                                                                       \
        cudssStatus_t status = call;                                           \
        if (status != CUDSS_STATUS_SUCCESS) {                                  \
            throw std::runtime_error(std::string("cuDSS error: ") +            \
                                     std::to_string(static_cast<int>(status)));\
        }                                                                      \
    } while (0)
#endif

/**
 * Convert COO format to CSR format on CPU
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> coo_to_csr_cpu(
    torch::Tensor row_indices,
    torch::Tensor col_indices,
    torch::Tensor values,
    int64_t num_rows,
    int64_t num_cols
) {
    // Move to CPU for conversion
    auto row_cpu = row_indices.to(torch::kCPU).contiguous();
    auto col_cpu = col_indices.to(torch::kCPU).contiguous();
    auto val_cpu = values.to(torch::kCPU).contiguous();
    
    int64_t nnz = values.size(0);
    
    // Get accessors
    int64_t* row_ptr = row_cpu.data_ptr<int64_t>();
    int64_t* col_ptr = col_cpu.data_ptr<int64_t>();
    
    // Create index array and sort by row, then column
    std::vector<int64_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        if (row_ptr[a] != row_ptr[b]) return row_ptr[a] < row_ptr[b];
        return col_ptr[a] < col_ptr[b];
    });
    
    // Apply permutation
    auto perm_tensor = torch::from_blob(perm.data(), {nnz}, torch::kInt64).clone();
    auto row_sorted = row_cpu.index_select(0, perm_tensor);
    auto col_sorted = col_cpu.index_select(0, perm_tensor);
    auto val_sorted = val_cpu.index_select(0, perm_tensor);
    
    // Build CSR row pointers
    auto csr_row_ptr = torch::zeros({num_rows + 1}, torch::kInt32);
    auto csr_col_ind = col_sorted.to(torch::kInt32);
    
    // Count elements in each row
    auto row_sorted_acc = row_sorted.accessor<int64_t, 1>();
    auto row_ptr_acc = csr_row_ptr.accessor<int, 1>();
    
    for (int64_t i = 0; i < nnz; i++) {
        row_ptr_acc[row_sorted_acc[i] + 1]++;
    }
    
    // Cumulative sum
    for (int64_t i = 1; i <= num_rows; i++) {
        row_ptr_acc[i] += row_ptr_acc[i - 1];
    }
    
    return std::make_tuple(csr_row_ptr, csr_col_ind, val_sorted);
}

#if CUDSS_AVAILABLE

/**
 * Solve sparse linear system Ax = b using cuDSS
 */
torch::Tensor cudss_solve(
    torch::Tensor indices,
    torch::Tensor values,
    int64_t m,
    int64_t n,
    torch::Tensor b,
    const std::string& matrix_type = "general",
    const std::string& reorder = "default"
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be on CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA");
    TORCH_CHECK(m == n, "Matrix must be square");
    TORCH_CHECK(values.scalar_type() == torch::kFloat64 || 
                values.scalar_type() == torch::kFloat32,
                "Only float32 and float64 are supported");
    
    int64_t nnz = values.size(0);
    int64_t nrhs = (b.dim() == 1) ? 1 : b.size(1);
    auto device = values.device();
    
    // Extract row and column indices
    auto row_indices = indices.select(0, 0);
    auto col_indices = indices.select(0, 1);
    
    // Convert to CSR format on CPU
    auto [csr_row_ptr_cpu, csr_col_ind_cpu, csr_values_cpu] = coo_to_csr_cpu(
        row_indices, col_indices, values, m, n
    );
    
    // Move to GPU
    auto csr_row_ptr = csr_row_ptr_cpu.to(device);
    auto csr_col_ind = csr_col_ind_cpu.to(device);
    auto csr_values = csr_values_cpu.to(values.scalar_type()).to(device);
    
    // Ensure b is contiguous and 2D for cuDSS
    auto b_2d = b.dim() == 1 ? b.unsqueeze(1) : b;
    b_2d = b_2d.contiguous();
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Create cuDSS handle
    cudssHandle_t handle;
    CHECK_CUDSS(cudssCreate(&handle));
    CHECK_CUDSS(cudssSetStream(handle, stream));
    
    // Determine data type
    cudaDataType_t cuda_dtype;
    if (values.scalar_type() == torch::kFloat64) {
        cuda_dtype = CUDA_R_64F;
    } else {
        cuda_dtype = CUDA_R_32F;
    }
    
    // Determine matrix type and view
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    
    if (matrix_type == "symmetric" || matrix_type == "SYMMETRIC") {
        mtype = CUDSS_MTYPE_SYMMETRIC;
        mview = CUDSS_MVIEW_LOWER;
    } else if (matrix_type == "spd" || matrix_type == "SPD") {
        mtype = CUDSS_MTYPE_SPD;
        mview = CUDSS_MVIEW_LOWER;
    }
    
    // Create matrix descriptor using new API (cuDSS 0.7+)
    cudssMatrix_t A_desc;
    CHECK_CUDSS(cudssMatrixCreateCsr(
        &A_desc,
        m, n, nnz,
        csr_row_ptr.data_ptr<int>(),
        nullptr,  // rowEnd (optional for standard CSR)
        csr_col_ind.data_ptr<int>(),
        csr_values.data_ptr(),
        CUDA_R_32I,    // index type (int32)
        cuda_dtype,    // value type
        mtype,         // matrix type
        mview,         // matrix view
        CUDSS_BASE_ZERO  // index base
    ));
    
    // Create dense matrix descriptors for b and x
    auto x = torch::zeros_like(b_2d);
    
    cudssMatrix_t b_desc, x_desc;
    CHECK_CUDSS(cudssMatrixCreateDn(
        &b_desc,
        m, nrhs,
        m,  // leading dimension
        b_2d.data_ptr(),
        cuda_dtype,
        CUDSS_LAYOUT_COL_MAJOR
    ));
    CHECK_CUDSS(cudssMatrixCreateDn(
        &x_desc,
        m, nrhs,
        m,
        x.data_ptr(),
        cuda_dtype,
        CUDSS_LAYOUT_COL_MAJOR
    ));
    
    // Create cuDSS config
    cudssConfig_t config;
    CHECK_CUDSS(cudssConfigCreate(&config));
    
    // Create cuDSS data holder
    cudssData_t data;
    CHECK_CUDSS(cudssDataCreate(handle, &data));
    
    // Analysis phase
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, 
                             A_desc, x_desc, b_desc));
    
    // Factorization phase
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data,
                             A_desc, x_desc, b_desc));
    
    // Solve phase
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data,
                             A_desc, x_desc, b_desc));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Cleanup
    CHECK_CUDSS(cudssDataDestroy(handle, data));
    CHECK_CUDSS(cudssConfigDestroy(config));
    CHECK_CUDSS(cudssMatrixDestroy(x_desc));
    CHECK_CUDSS(cudssMatrixDestroy(b_desc));
    CHECK_CUDSS(cudssMatrixDestroy(A_desc));
    CHECK_CUDSS(cudssDestroy(handle));
    
    // Return in original shape
    return b.dim() == 1 ? x.squeeze(1) : x;
}

torch::Tensor cudss_lu(torch::Tensor indices, torch::Tensor values, int64_t m, int64_t n, torch::Tensor b) {
    return cudss_solve(indices, values, m, n, b, "general", "default");
}

torch::Tensor cudss_cholesky(torch::Tensor indices, torch::Tensor values, int64_t m, int64_t n, torch::Tensor b) {
    return cudss_solve(indices, values, m, n, b, "spd", "default");
}

torch::Tensor cudss_ldlt(torch::Tensor indices, torch::Tensor values, int64_t m, int64_t n, torch::Tensor b) {
    return cudss_solve(indices, values, m, n, b, "symmetric", "default");
}

#else

// Stub implementations when cuDSS is not available
torch::Tensor cudss_solve(torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, const std::string&, const std::string&) {
    throw std::runtime_error("cuDSS is not available. Please install the NVIDIA cuDSS library.");
}

torch::Tensor cudss_lu(torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor) {
    throw std::runtime_error("cuDSS is not available. Please install the NVIDIA cuDSS library.");
}

torch::Tensor cudss_cholesky(torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor) {
    throw std::runtime_error("cuDSS is not available. Please install the NVIDIA cuDSS library.");
}

torch::Tensor cudss_ldlt(torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor) {
    throw std::runtime_error("cuDSS is not available. Please install the NVIDIA cuDSS library.");
}

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &cudss_solve, 
          "Solve sparse linear system using cuDSS",
          py::arg("indices"), py::arg("values"), py::arg("m"), py::arg("n"),
          py::arg("b"), py::arg("matrix_type") = "general", 
          py::arg("reorder") = "default");
    m.def("lu", &cudss_lu, 
          "Solve sparse linear system using cuDSS LU factorization",
          py::arg("indices"), py::arg("values"), py::arg("m"), py::arg("n"),
          py::arg("b"));
    m.def("cholesky", &cudss_cholesky,
          "Solve sparse linear system using cuDSS Cholesky factorization",
          py::arg("indices"), py::arg("values"), py::arg("m"), py::arg("n"),
          py::arg("b"));
    m.def("ldlt", &cudss_ldlt,
          "Solve sparse linear system using cuDSS LDLT factorization",
          py::arg("indices"), py::arg("values"), py::arg("m"), py::arg("n"),
          py::arg("b"));
}
