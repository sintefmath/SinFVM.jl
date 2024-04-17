using SinSWE
using StaticArrays
using Test
import CUDA

function runonbackend(backend, grid, numericalflux, input_eval_equation, output_eval_upwind)

end


function runonbackend(backend::SinSWE.CPUBackend, grid, numericalflux, input_eval_equation, output_eval_upwind)
    for index in 2:grid.totalcells[1] - 1
        r = index + 1
        l = index - 1
        output_eval_upwind[index], wavespeed = numericalflux(input_eval_equation[r], input_eval_equation[l], XDIR)
    end
end



#backend = make_cuda_backend()
for backend in SinSWE.get_available_backends() 

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .*0, 0.0 .* x]
    nx = 8
    grid = SinSWE.CartesianGrid(nx; gc=2)
    equation = SinSWE.ShallowWaterEquations1D()
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

    runonbackend(backend, grid, numericalflux, input_eval_equation, output_eval_equation)
    
    # Test flux
    SinSWE.@fvmloop SinSWE.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_upwind[index], dontusethis = numericalflux(input_eval_equation[r], input_eval_equation[l], XDIR)
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

