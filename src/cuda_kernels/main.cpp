#include <torch/extension.h>

torch::Tensor forward_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor forward_global_coalesce(torch::Tensor A, torch::Tensor B);
torch::Tensor forward_shared(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_naive", torch::wrap_pybind_function(forward_naive), "forward_naive");
    m.def("forward_global_coalesce", torch::wrap_pybind_function(forward_global_coalesce), "forward_global_coalesce");
    m.def("forward_shared", torch::wrap_pybind_function(forward_shared), "forward_shared");
}