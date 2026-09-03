# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using VolumeFluxes
isdefined(Main, :test_backends) || include("testing_utils.jl")
using StaticArrays
using Test
import CUDA

function runonbackend(backend, grid, numericalflux, input_eval_equation, output_eval_upwind)

end


function runonbackend(backend::VolumeFluxes.CPUBackend, grid, numericalflux, input_eval_equation, output_eval_upwind)
    for index in 2:grid.totalcells[1] - 1
        r = index + 1
        l = index - 1
        output_eval_upwind[index], wavespeed = numericalflux(input_eval_equation[r], input_eval_equation[l], XDIR)
    end
end



#backend = make_cuda_backend()
@testset "$(backend_label(backend))" for backend in test_backends() 

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .*0, 0.0 .* x]
    nx = 8
    grid = VolumeFluxes.CartesianGrid(nx; gc=2)
    equation = backend_params(backend, VolumeFluxes.ShallowWaterEquations1D())
    output_eval_equation = VolumeFluxes.Volume(backend, equation, grid)


    input_eval_equation = VolumeFluxes.Volume(backend, equation, grid)# .+ 1
    CUDA.@allowscalar VolumeFluxes.InteriorVolume(input_eval_equation)[:] = u0.(VolumeFluxes.cell_centers(grid))

    ## Test equation on inner cells
    VolumeFluxes.@fvmloop VolumeFluxes.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_equation[index] = equation(XDIR, input_eval_equation[index]...)
    end

    # Test equation and eigenvalues on all cells
    VolumeFluxes.@fvmloop VolumeFluxes.for_each_cell(backend, grid) do index
        output_eval_equation[index] = equation(XDIR, input_eval_equation[index]...)
        ignored = VolumeFluxes.compute_eigenvalues(equation, XDIR, input_eval_equation[index]...)
    end

    output_eval_upwind = VolumeFluxes.Volume(backend, equation, grid)
    numericalflux = VolumeFluxes.CentralUpwind(equation)

    runonbackend(backend, grid, numericalflux, input_eval_equation, output_eval_equation)
    
    # Test flux
    VolumeFluxes.@fvmloop VolumeFluxes.for_each_inner_cell(backend, grid, XDIR) do l, index, r
        output_eval_upwind[index], dontusethis = numericalflux(input_eval_equation[r], input_eval_equation[l], XDIR)
    end


    output_eval_recon_l = VolumeFluxes.Volume(backend, equation, grid)
    output_eval_recon_r = VolumeFluxes.Volume(backend, equation, grid)
    linrec = VolumeFluxes.LinearReconstruction()

    # Test reconstruction
    # VolumeFluxes.reconstruct!(backend, linrec, output_eval_recon_l, output_eval_recon_r, input_eval_equation, grid, equation, XDIR)


    h = collect(VolumeFluxes.InteriorVolume(output_eval_upwind).h)
    hu = collect(VolumeFluxes.InteriorVolume(output_eval_upwind).hu)
 

    @show h
    @show hu
    #@test all(! . isnan.(h))
    #@test all(!. isnan.(hu))
end
# linrec = VolumeFluxes.LinearReconstruction(1.05)
# numericalflux = VolumeFluxes.CentralUpwind(equation)
# timestepper = VolumeFluxes.ForwardEulerStepper()
# conserved_system = VolumeFluxes.ConservedSystem(backend, linrec, numericalflux, equation, grid)
