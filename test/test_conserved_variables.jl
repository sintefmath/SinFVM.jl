using SinSWE
using CUDA

N = 10
x = ConservedVariables(zeros(N), zeros(N), zeros(N))

xcu = cu(x)

@show xcu
@show typeof(xcu)