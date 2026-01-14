/**
 * cuSOLVER sparse solver backend for torch-sla
 * 
 * This file implements sparse linear equation solvers using NVIDIA cuSOLVER library.
 * Supports:
 *   - LU decomposition (cusolverSpDcsrlsvlu)
 *   - Cholesky decomposition (cusolverSpDcsrlsvchol) for SPD matrices
 *   - QR decomposition (cusolverSpDcsrlsvqr)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>

// Error checking macros
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error at ") +           \
                                     __FILE__ + ":" + std::to_string(__LINE__) + \
                                     " - " + cudaGetErrorString(err));         \
        }                                                                      \
    } while (0)

#define CHECK_CUSOLVER(call)                                                   \
    do {                                                                       \
        cusolverStatus_t status = call;                                        \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            throw std::runtime_error(std::string("cuSOLVER error: ") +         \
                                     std::to_string(status));                  \
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

/**
 * Solve sparse linear system Ax = b using cuSOLVER QR decomposition
 */
torch::Tensor cusolver_qr(
    torch::Tensor indices,
    torch::Tensor values,
    int64_t m,
    int64_t n,
    torch::Tensor b,
    double tol
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be on CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA");
    TORCH_CHECK(m == n, "Matrix must be square for QR solver");
    TORCH_CHECK(values.scalar_type() == torch::kFloat64, "Only float64 is supported");
    
    int64_t nnz = values.size(0);
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
    auto csr_values = csr_values_cpu.to(torch::kFloat64).to(device);
    auto b_contig = b.contiguous();
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Create cuSOLVER handle
    cusolverSpHandle_t cusolverH = nullptr;
    cusparseMatDescr_t descrA = nullptr;
    
    CHECK_CUSOLVER(cusolverSpCreate(&cusolverH));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    
    // Allocate output
    auto x = torch::zeros_like(b);
    int singularity = -1;
    
    // Solve using QR
    CHECK_CUSOLVER(cusolverSpDcsrlsvqr(
        cusolverH,
        m,
        nnz,
        descrA,
        csr_values.data_ptr<double>(),
        csr_row_ptr.data_ptr<int>(),
        csr_col_ind.data_ptr<int>(),
        b_contig.data_ptr<double>(),
        tol,
        0,  // reorder (0 = no reorder for device)
        x.data_ptr<double>(),
        &singularity
    ));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    if (singularity >= 0) {
        std::cerr << "Warning: Matrix is singular at row " << singularity << std::endl;
    }
    
    // Cleanup
    if (descrA) cusparseDestroyMatDescr(descrA);
    if (cusolverH) cusolverSpDestroy(cusolverH);
    
    return x;
}

/**
 * Solve sparse linear system Ax = b using cuSOLVER Cholesky decomposition
 * For symmetric positive definite matrices
 */
torch::Tensor cusolver_cholesky(
    torch::Tensor indices,
    torch::Tensor values,
    int64_t m,
    int64_t n,
    torch::Tensor b,
    double tol
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be on CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA");
    TORCH_CHECK(m == n, "Matrix must be square for Cholesky solver");
    TORCH_CHECK(values.scalar_type() == torch::kFloat64, "Only float64 is supported");
    
    int64_t nnz = values.size(0);
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
    auto csr_values = csr_values_cpu.to(torch::kFloat64).to(device);
    auto b_contig = b.contiguous();
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Create cuSOLVER handle
    cusolverSpHandle_t cusolverH = nullptr;
    cusparseMatDescr_t descrA = nullptr;
    
    CHECK_CUSOLVER(cusolverSpCreate(&cusolverH));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    
    // Allocate output
    auto x = torch::zeros_like(b);
    int singularity = -1;
    
    // Solve using Cholesky
    CHECK_CUSOLVER(cusolverSpDcsrlsvchol(
        cusolverH,
        m,
        nnz,
        descrA,
        csr_values.data_ptr<double>(),
        csr_row_ptr.data_ptr<int>(),
        csr_col_ind.data_ptr<int>(),
        b_contig.data_ptr<double>(),
        tol,
        0,  // reorder
        x.data_ptr<double>(),
        &singularity
    ));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    if (singularity >= 0) {
        std::cerr << "Warning: Matrix is not positive definite at row " << singularity << std::endl;
    }
    
    // Cleanup
    if (descrA) cusparseDestroyMatDescr(descrA);
    if (cusolverH) cusolverSpDestroy(cusolverH);
    
    return x;
}

/**
 * Solve sparse linear system Ax = b using cuSOLVER LU decomposition (host-based)
 */
torch::Tensor cusolver_lu(
    torch::Tensor indices,
    torch::Tensor values,
    int64_t m,
    int64_t n,
    torch::Tensor b,
    double tol
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be on CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA");
    TORCH_CHECK(m == n, "Matrix must be square for LU solver");
    TORCH_CHECK(values.scalar_type() == torch::kFloat64, "Only float64 is supported");
    
    int64_t nnz = values.size(0);
    auto device = values.device();
    
    // Extract row and column indices
    auto row_indices = indices.select(0, 0);
    auto col_indices = indices.select(0, 1);
    
    // Convert to CSR format on CPU
    auto [csr_row_ptr_cpu, csr_col_ind_cpu, csr_values_cpu] = coo_to_csr_cpu(
        row_indices, col_indices, values, m, n
    );
    
    csr_values_cpu = csr_values_cpu.to(torch::kFloat64);
    auto b_cpu = b.contiguous().to(torch::kCPU);
    auto x_cpu = torch::zeros_like(b_cpu);
    
    // Create cuSOLVER handle
    cusolverSpHandle_t cusolverH = nullptr;
    cusparseMatDescr_t descrA = nullptr;
    
    CHECK_CUSOLVER(cusolverSpCreate(&cusolverH));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    
    int singularity = -1;
    
    // LU solver only works on host
    CHECK_CUSOLVER(cusolverSpDcsrlsvluHost(
        cusolverH,
        m,
        nnz,
        descrA,
        csr_values_cpu.data_ptr<double>(),
        csr_row_ptr_cpu.data_ptr<int>(),
        csr_col_ind_cpu.data_ptr<int>(),
        b_cpu.data_ptr<double>(),
        tol,
        1,  // reorder using symrcm
        x_cpu.data_ptr<double>(),
        &singularity
    ));
    
    if (singularity >= 0) {
        std::cerr << "Warning: Matrix is singular at row " << singularity << std::endl;
    }
    
    // Copy result back to GPU
    auto x = x_cpu.to(device);
    
    // Cleanup
    if (descrA) cusparseDestroyMatDescr(descrA);
    if (cusolverH) cusolverSpDestroy(cusolverH);
    
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qr", &cusolver_qr, "Solve sparse linear system using cuSOLVER QR",
          py::arg("indices"), py::arg("values"), py::arg("m"), py::arg("n"),
          py::arg("b"), py::arg("tol") = 1e-12);
    m.def("cholesky", &cusolver_cholesky, "Solve sparse linear system using cuSOLVER Cholesky",
          py::arg("indices"), py::arg("values"), py::arg("m"), py::arg("n"),
          py::arg("b"), py::arg("tol") = 1e-12);
    m.def("lu", &cusolver_lu, "Solve sparse linear system using cuSOLVER LU",
          py::arg("indices"), py::arg("values"), py::arg("m"), py::arg("n"),
          py::arg("b"), py::arg("tol") = 1e-12);
}
