using SinFVM
using StaticArrays
using Test
import CUDA

function runonbackend(backend, grid, numericalflux, input_eval_equation, output_eval_upwind)

end


function runonbackend(backend::SinFVM.CPUBackend, grid, numericalflux, input_eval_equation, output_eval_upwind)
    for index in 2:grid.totalcells[1] - 1
        r = index + 1
        l = index - 1
        output_eval_upwind[index], wavespeed = numericalflux(input_eval_equation[r], input_eval_equation[l], XDIR)
    end
end



#backend = make_cuda_backend()
for backend in SinFVM.get_available_backends() 

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .*0, 0.0 .* x]
    nx = 8
    grid = SinFVM.CartesianGrid(nx; gc=2)
    equation = SinFVM.ShallowWaterEquations1D()
    output_eval_equation = SinFVM.Volume(backend, equation, grid)


    input_eval_equation = SinFVM.Volume(backend, equation, grid)# .+ 1
    CUDA.@allowscalar SinFVM.InteriorVolume(input_eval_equation)[:] = u0.(SinFVM.cell_centers(grid))

    ## Test equation on inner cells
    SinFVM.@fvmloop SinFVM.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_equation[index] = equation(XDIR, input_eval_equation[index]...)
    end

    # Test equation and eigenvalues on all cells
    SinFVM.@fvmloop SinFVM.for_each_cell(backend, grid) do index
        output_eval_equation[index] = equation(XDIR, input_eval_equation[index]...)
        ignored = SinFVM.compute_eigenvalues(equation, XDIR, input_eval_equation[index]...)
    end

    output_eval_upwind = SinFVM.Volume(backend, equation, grid)
    numericalflux = SinFVM.CentralUpwind(equation)

    runonbackend(backend, grid, numericalflux, input_eval_equation, output_eval_equation)
    
    # Test flux
    SinFVM.@fvmloop SinFVM.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_upwind[index], dontusethis = numericalflux(input_eval_equation[r], input_eval_equation[l], XDIR)
    end


    output_eval_recon_l = SinFVM.Volume(backend, equation, grid)
    output_eval_recon_r = SinFVM.Volume(backend, equation, grid)
    linrec = SinFVM.LinearReconstruction()

    # Test reconstruction
    # SinFVM.reconstruct!(backend, linrec, output_eval_recon_l, output_eval_recon_r, input_eval_equation, grid, equation, XDIR)


    h = collect(SinFVM.InteriorVolume(output_eval_upwind).h)
    hu = collect(SinFVM.InteriorVolume(output_eval_upwind).hu)
 

    @show h
    @show hu
    #@test all(! . isnan.(h))
    #@test all(!. isnan.(hu))
end
# linrec = SinFVM.LinearReconstruction(1.05)
# numericalflux = SinFVM.CentralUpwind(equation)
# timestepper = SinFVM.ForwardEulerStepper()
# conserved_system = SinFVM.ConservedSystem(backend, linrec, numericalflux, equation, grid)

