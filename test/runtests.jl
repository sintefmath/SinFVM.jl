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

using SinFVM
using Test

# TODO: Go through tests and check they do not take longer time than necessary
using CairoMakie

# Disable showing the plot in CairoMakie
CairoMakie.activate!(type = "svg")
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
