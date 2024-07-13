#include <torch/extension.h>
#include <torch/torch.h>
#include <cmath>
#include <vector>


torch::Tensor coo_diagonal(torch::Tensor A, double at_least = 1.) {
    TORCH_CHECK(A.is_sparse(), "A should be sparse");
    TORCH_CHECK(A.dim() == 2, "A should be 2 dimensional");
    TORCH_CHECK(A.size(0) == A.size(1), "A should be square");

    auto indices = A._indices();
    auto values = A._values();

    auto N = A.size(0);

    auto mask = indices.select(0, 0) == indices.select(0, 1);
    auto cand_values = values.masked_select(mask);
    auto cand_indices = indices.select(0, 0).masked_select(mask);

    auto diag_values = torch::full({N}, at_least, cand_values.options());
    //auto diag_values = torch::zeros({N}, cand_values.options());
    diag_values.index_put_({cand_indices}, cand_values);

    return diag_values;
}


torch::Tensor jacobi_precond(
    torch::Tensor A
){
    auto x = coo_diagonal(A, 1.0).reciprocal_();
    return x;
}


torch::Tensor bicgstab(
    torch::Tensor indices, 
    torch::Tensor values, 
    int m, int n,
    torch::Tensor b, 
    double atol = 1e-5, 
    int64_t max_iter = 10000) {
    // https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    TORCH_CHECK(indices.dim() == 2, "Indices must be two-dimensional.");
    TORCH_CHECK(values.dim() == 1, "Values must be one-dimensional.");
    TORCH_CHECK(indices.size(1) == values.size(0), "Indices and values must have same size in first dimension.");
    TORCH_CHECK(b.dim() == 1, "Vector b must be one-dimensional.");
    TORCH_CHECK(b.size(0) == m, "Vector b must have same size as first dimension of A.");
    TORCH_CHECK(m == n, "A must be square.");
  
    torch::TensorOptions options;
    torch::Tensor A;

    try{
        options = values.options().layout(torch::kSparse);
        A = torch::sparse_coo_tensor(indices, values, {m, n}, options);
    }catch (const c10::Error& e) {
        std::cerr << "spsolve.cpp : torch::sparse_coo_tensor creation fail:" << e.msg() << std::endl;
        std::exit(EXIT_FAILURE);
    }
   

    // jacobi preconditioner https://en.wikipedia.org/wiki/Preconditioner
    // auto x0 = coo_diagonal(A, 1.0).reciprocal_(); // Assuming coo_diagonal() is implemented

    try{
        A = A.to_sparse_csr();
    }catch (const c10::Error& e) {
        std::cerr << "spsolve.cpp : A.to_sparse_csr() fail:" << e.msg() << std::endl;
        std::exit(EXIT_FAILURE);
    }
   
    torch::Tensor x;
    torch::Tensor r;
    torch::Tensor r0_hat;
    torch::Tensor v;
    torch::Tensor p;
    torch::Tensor h;
    torch::Tensor s;
    torch::Tensor t;
    double rho = 1.0;
    double rho_new = 1.0;
    double alpha = 1.0;
    double beta = 1.0;
    double omega = 1.0;

    try{
        x = torch::zeros_like(b);
        r = b - torch::mv(A, x);
        r0_hat = r.clone();
        v = r.clone();
        p = r.clone();
        h = torch::zeros_like(b);
        s = torch::zeros_like(b);
        t = torch::zeros_like(b);
        
    }catch (const c10::Error& e) {
        std::cerr << "spsolve.cpp : initialization fail:" << e.msg() << std::endl;
        std::exit(EXIT_FAILURE);
    }
   

    for (int64_t i = 0; i < max_iter; ++i) {

        // std::cout<<"iteration "<<i<<std::endl;

        v = torch::mv(A, p);
        alpha = rho / torch::dot(r0_hat, v).item<double>();
        h = x + alpha * p;
        s = r - alpha * v;

        if (torch::norm(s).item<double>() < atol) {
            return h;
        }

        t = torch::mv(A, s);
        omega = torch::dot(t, s).item<double>() / torch::dot(t, t).item<double>();
        x = h + omega * s;

        r = s - omega * t;

        if (torch::norm(r).item<double>() < atol) {
            return x;
        }

        rho_new = torch::dot(r0_hat, r).item<double>();
        beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
        p = r + beta * (p - omega * v);
    }
    
    std::cerr << "bicgstab did not converge after " << max_iter << " iterations." << std::endl;

    return x;
}


torch::Tensor cg(
    torch::Tensor indices, 
    torch::Tensor values, 
    int m, int n,
    torch::Tensor b, 
    double atol = 1e-5, 
    int64_t max_iter = 10000) {
    // https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    TORCH_CHECK(indices.dim() == 2, "Indices must be two-dimensional.");
    TORCH_CHECK(values.dim() == 1, "Values must be one-dimensional.");
    TORCH_CHECK(indices.size(1) == values.size(0), "Indices and values must have same size in first dimension.");
    TORCH_CHECK(b.dim() == 1, "Vector b must be one-dimensional.");
    TORCH_CHECK(b.size(0) == m, "Vector b must have same size as first dimension of A.");
    TORCH_CHECK(m == n, "A must be square.");
  
    auto options = values.options().layout(torch::kSparse);
    auto A = torch::sparse_coo_tensor(indices, values, {m, n}, options);

    // jacobi preconditioner https://en.wikipedia.org/wiki/Preconditioner
    // auto x0 = coo_diagonal(A, 1.0).reciprocal_(); // Assuming coo_diagonal() is implemented
    auto x0 = torch::zeros_like(b);

    A = A.to_sparse_csr();

    auto r = b - torch::mv(A, x0);
    auto p = r.clone();
    auto x = x0;
    double rs_old = torch::dot(r, r).item<double>();

    for (int64_t i = 0; i < max_iter; ++i) {

        auto Ap = torch::mv(A, p);
        double alpha = rs_old / torch::dot(p, Ap).item<double>();
        x = x + alpha * p;
        r = r - alpha * Ap;
        double rs_new = torch::dot(r, r).item<double>();

        if (sqrt(rs_new) < atol) {
            return x;
        }

        p = r + (rs_new / rs_old) * p;
        rs_old = rs_new;
    }
    
    std::cerr << "bicgstab did not converge after " << max_iter << " iterations." << std::endl;

    return x;
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bicgstab", &bicgstab, "bicgstab");
  m.def("cg", &cg, "cg");
  m.def("coo_diagonal", &coo_diagonal, "coo_diagonal");
}