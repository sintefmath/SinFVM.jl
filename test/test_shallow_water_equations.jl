using SinSWE
using StaticArrays
using Test
import CUDA

u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .*0, 0.0 .* x]
nx = 8
grid = SinSWE.CartesianGrid(nx; gc=2)

#backend = make_cuda_backend()
for backend in SinSWE.get_available_backends() # make_cpu_backend()
    @show backend
    equation = SinSWE.ShallowWaterEquations1D(backend, grid)
    output_eval_equation = SinSWE.Volume(backend, equation, grid)


    input_eval_equation = SinSWE.Volume(backend, equation, grid)# .+ 1
    CUDA.@allowscalar SinSWE.InteriorVolume(input_eval_equation)[:] = u0.(SinSWE.cell_centers(grid))

    ## Test equation on inner cells
    SinSWE.@fvmloop SinSWE.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_equation[index] = equation(XDIR, input_eval_equation[index]...)
    end

    # Test equation and eigenvalues on all cells
    SinSWE.@fvmloop SinSWE.for_each_cell(backend, grid) do index
        output_eval_equation[index] = equation(XDIR, input_eval_equation[index]...)
        ignored = SinSWE.compute_eigenvalues(equation, XDIR, input_eval_equation[index]...)
    end

    output_eval_upwind = SinSWE.Volume(backend, equation, grid)
    numericalflux = SinSWE.CentralUpwind(equation)

    # Test flux
    SinSWE.@fvmloop SinSWE.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_upwind[index], wavespeed = numericalflux(input_eval_equation[r], input_eval_equation[l])
    end


    output_eval_recon_l = SinSWE.Volume(backend, equation, grid)
    output_eval_recon_r = SinSWE.Volume(backend, equation, grid)
    linrec = SinSWE.LinearReconstruction()

    # Test reconstruction
    # SinSWE.reconstruct!(backend, linrec, output_eval_recon_l, output_eval_recon_r, input_eval_equation, grid, equation, XDIR)


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

