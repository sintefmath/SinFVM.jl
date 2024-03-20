using SinSWE
using StaticArrays
using Test
import CUDA

u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5, 0.0 .* x]
nx = 8
grid = SinSWE.CartesianGrid(nx; gc=2)

#backend = make_cuda_backend()
for backend in SinSWE.get_available_backends() # make_cpu_backend()
    @show backend
    equation = SinSWE.ShallowWaterEquations1D(backend, grid)
    output_eval_equation = SinSWE.Volume(backend, equation, grid)


    input_eval_equation = SinSWE.Volume(backend, equation, grid)# .+ 1
    CUDA.@allowscalar SinSWE.InteriorVolume(input_eval_equation)[:] = u0.(SinSWE.cell_centers(grid))
    SinSWE.@fvmloop SinSWE.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_equation[index] = equation(XDIR, input_eval_equation[index]...)
    end

    SinSWE.@fvmloop SinSWE.for_each_cell(backend, grid) do index
        output_eval_equation[index] = equation(XDIR, input_eval_equation[index]...)
        ignored = SinSWE.compute_eigenvalues(equation, XDIR, input_eval_equation[index]...)
    end

    output_eval_upwind = SinSWE.Volume(backend, equation, grid)
    numericalflux = SinSWE.CentralUpwind(equation)

    SinSWE.@fvmloop SinSWE.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_upwind[index], wavespeed = numericalflux(input_eval_equation[r], input_eval_equation[l])
    end

    h = collect(SinSWE.InteriorVolume(output_eval_upwind).h)
    hu = collect(SinSWE.InteriorVolume(output_eval_upwind).hu)
    
    @show h
    @show hu
    #@test all(! . isnan.(h))
    #@test all(!. isnan.(hu))
end
# linrec = SinSWE.LinearReconstruction(1.05)
# numericalflux = SinSWE.CentralUpwind(equation)
# timestepper = SinSWE.ForwardEulerStepper()
# conserved_system = SinSWE.ConservedSystem(backend, linrec, numericalflux, equation, grid)

