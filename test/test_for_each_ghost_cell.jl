using SinSWE
using CUDA
using Test


nx = 10
grid = SinSWE.CartesianGrid(nx)
backend = make_cuda_backend()

x_d = CUDA.fill(1.0, nx)
y_d = CUDA.fill(2.0, nx)
SinSWE.@fvmloop SinSWE.for_each_ghost_cell(backend, grid, XDIR) do i
    y_d[i] = x_d[i] - i
end
y = collect(y_d)
@test y[1] == 0.0
@test y[2:end] == 2.0 * ones(nx - 1)