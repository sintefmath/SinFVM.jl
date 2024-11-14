using SinFVM
using Test

# TODO: Go through tests and check they do not take longer time than necessary

@testset "SinFVM tests" begin
    # Run all scripts in test/test_*.jl
    ls_test = readdir("test")
    for test_file in readdir("test")
        if startswith(test_file, "test_") && endswith(test_file, ".jl")
            #@show test_name, test_file
            
            test_name = replace(test_file, ".jl"=>"")
            @testset "$(test_name)" begin
                include(test_file)
            end
      
        end
    end
end
nothing