# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using StaticArrays
using LinearAlgebra
using Test
import CUDA

using SinFVM


function run_sim(backend, rain, T_hours, grid; include_momentum_test=true)
    equation = SinFVM.ShallowWaterEquations()
    reconstruction = SinFVM.LinearReconstruction()
    numericalflux = SinFVM.CentralUpwind(equation)

    conserved_system =
        SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, rain)
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid)

    u0 = x -> @SVector[1.0, 0.0, 0.0]
    x = SinFVM.cell_centers(grid)
    initial = u0.(x)
    SinFVM.set_current_state!(simulator, initial)

    @time SinFVM.simulate_to_time(simulator, T_hours*3600)
    @test SinFVM.current_time(simulator) ≈ T_hours*3600 atol=1e-10

    results = SinFVM.current_interior_state(simulator)
    if include_momentum_test
        @test all(collect(results.hu) .== 0.0)
        @test all(collect(results.hv) .== 0.0)
    end
    return collect(results.h)
end

constant_rain = SinFVM.ConstantRain(0.01)
@test constant_rain.rain_rate == 0.01
@test SinFVM.compute_rain(constant_rain, 12.6, CartesianIndex(12, 56)) == 0.01/3600.0
@test SinFVM.compute_rain(constant_rain, 12.6) == 0.01/3600.0
@test SinFVM.compute_rain(constant_rain) == 0.01/3600.0

time_dependent_rain = SinFVM.TimeDependentRain(@SVector[15.0, 0.3, 20.0].*0.001, @SVector[0.0, 1.0, 2.0].*3600)
@test SinFVM.compute_rain(time_dependent_rain, 10)  == 0.015/3600.0
@test SinFVM.compute_rain(time_dependent_rain, 3610)  == 0.0003/3600.0
@test SinFVM.compute_rain(time_dependent_rain, 7210)  == 0.02/3600.0
@test SinFVM.compute_rain(time_dependent_rain, 1.0e6)  == 0.02/3600.0
@test SinFVM.compute_rain(time_dependent_rain, 0.0)  == 0.015/3600.0

grid1 = SinFVM.CartesianGrid(10,  10;  gc=2, boundary=SinFVM.WallBC(), extent=[0.0 1000.0; 0.0 1000.0])
grid2 = SinFVM.CartesianGrid(100, 100; gc=2, boundary=SinFVM.WallBC(), extent=[0.0 1000.0; 0.0 1000.0])

h1 = run_sim(SinFVM.make_cpu_backend(), constant_rain, 3, grid1)
@test maximum(h1) ≈ (3*0.01 + 1.0) atol=1e-12
@test minimum(h1) ≈ (3*0.01 + 1.0) atol=1e-12
if SinFVM.has_cuda_backend()
    h2 = run_sim(SinFVM.make_cuda_backend(), constant_rain, 3, grid2)
    @test maximum(h2) ≈ (3*0.01 + 1.0) atol=1e-12
    @test minimum(h2) ≈ (3*0.01 + 1.0) atol=1e-12
end
# Check time dependent rain
# Comment: We get the wrong rain in the time step where we change intensity.
#   This causes an error on the scale of rain_intensity*dt/3600
h3 = run_sim(SinFVM.make_cpu_backend(), time_dependent_rain, 1, grid1)
@test maximum(h3) ≈ (0.015 + 1.0) atol=1e-12
@test minimum(h3) ≈ (0.015 + 1.0) atol=1e-12
h4 = run_sim(SinFVM.make_cpu_backend(), time_dependent_rain, 2, grid1)
@test maximum(h4) ≈ (0.0153 + 1.0) atol=1e-4
@test minimum(h4) ≈ (0.0153 + 1.0) atol=1e-4
h5 = run_sim(SinFVM.make_cpu_backend(), time_dependent_rain, 3, grid1)
@test maximum(h5) ≈ (0.0353 + 1.0) atol=1e-4
@test minimum(h5) ≈ (0.0353 + 1.0) atol=1e-4

if SinFVM.has_cuda_backend()
    h6 = run_sim(SinFVM.make_cuda_backend(), time_dependent_rain, 3, grid2)
    @test maximum(h6) ≈ (0.0353 + 1.0) atol=1e-7
    @test minimum(h6) ≈ (0.0353 + 1.0) atol=1e-7
end

##### Tests for FunctionalRain
function constant_rain_function(t, x, y)
    return 0.01/3600.0
end
constant_functional_rain = SinFVM.FunctionalRain(constant_rain_function, grid1)
h1_f = run_sim(SinFVM.make_cpu_backend(), constant_functional_rain, 3, grid1)
@test maximum(h1_f) ≈ (3*0.01 + 1.0) atol=1e-12
@test minimum(h1_f) ≈ (3*0.01 + 1.0) atol=1e-12

if SinFVM.has_cuda_backend()
    h2_f = run_sim(SinFVM.make_cuda_backend(), constant_functional_rain, 3, grid1)
    @test maximum(h2_f) ≈ (3*0.01 + 1.0) atol=1e-12
    @test minimum(h2_f) ≈ (3*0.01 + 1.0) atol=1e-12
end
function timedependent_rain_function(t, x, y)
    rate = 0
    if t < 1*3600
        rate = 15.0*0.001
    elseif t < 2*3600
        rate = 0.3*0.001
    else
        rate = 20.0*0.001
    end
    return rate/3600
end
timedependent_functional_rain = SinFVM.FunctionalRain(timedependent_rain_function, grid1)
h3_f = run_sim(SinFVM.make_cpu_backend(), timedependent_functional_rain, 3, grid1)
@test maximum(h3_f) ≈ (0.0353 + 1.0) atol=1e-4
@test minimum(h3_f) ≈ (0.0353 + 1.0) atol=1e-4

if SinFVM.has_cuda_backend()
    h4_f = run_sim(SinFVM.make_cuda_backend(), timedependent_functional_rain, 3, grid2)
    @test maximum(h4_f) ≈ (0.0353 + 1.0) atol=1e-4
    @test minimum(h4_f) ≈ (0.0353 + 1.0) atol=1e-4
end
function spatial_rain_function(t, x, y)
    # @show t, x, y
    if t > 3600.0 && t < 7200.0
        return 3*(x/500)*(y/500)*0.01/3600.0
    else
        return 0.0
    end
end
spatial_functional_rain = SinFVM.FunctionalRain(spatial_rain_function, grid1)
h5_f = run_sim(SinFVM.make_cpu_backend(), spatial_functional_rain, 3, grid1, include_momentum_test=false)

if SinFVM.has_cuda_backend()
    @test sum(h5_f) ≈ sum(h2_f) atol=1e-2
end


# @show sum(collect(results1.h))*100*100
# @show sum(10.0*1000*1000)

println("success")
