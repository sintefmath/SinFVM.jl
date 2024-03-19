using SinSWE
using Test

@testset "SinSWE tests" begin
    @testset "Boundary condition tests" begin
        include("test_update_bc.jl")
    end
    @testset "Grid tests" begin
        include("test_grid.jl")
    end 
end