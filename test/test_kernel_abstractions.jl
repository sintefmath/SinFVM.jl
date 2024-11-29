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

using KernelAbstractions
import CUDA
using StaticArrays
using SinFVM

function get_variable_names_referred(expression)
    nothing
end


function get_variable_names_referred(expression::Symbol)
    return [expression]
end

function get_variable_names_referred(expression::Expr)
    if expression.head == :call
        return get_variable_names_referred(expression.args)
    else
        return get_variable_names_referred(expression.args)
    end
end

function get_variable_names_referred(expressions::Vector)
    all_symbols = []

    for expression in expressions
        symbols_from_expression = get_variable_names_referred(expression)

        if !isnothing(symbols_from_expression)
            all_symbols = vcat(all_symbols, symbols_from_expression)
        end
    end
    return all_symbols
end


macro for_each_cell_macro(code_snippet)
    modules = [mod for mod in getfield.(Ref(Main), names(Main)) if typeof(mod) == Module && mod != Main]
    all_functions = vcat((filter!(f -> true, #typeof(f), <: Function, 
        getfield.(Ref(m), names(m))) for m in modules)...)
    # @show (all_functions)
    # println("macro is called")
    # @show code_snippet
    #dump(code_snippet; maxdepth=18)
    # @show code_snippet.head
    # @show code_snippet.args
    # @show size(code_snippet.args)
    # @show code_snippet.args[2]
    parameter_names = get_variable_names_referred(code_snippet.args[2].args[1])
    function_body = code_snippet.args[2].args[2]

    all_variables_referenced = get_variable_names_referred(function_body.args)
    all_variables_referenced = Set(all_variables_referenced)

    for known_symbol in all_functions
        delete!(all_variables_referenced, Symbol(known_symbol))
    end
    #delete!(all_variables_referenced, :nothing)
    # @show all_variables_referenced
    # @show parameter_names

    new_parameter_names = []
    for variable_referred in all_variables_referenced
        if !(variable_referred in parameter_names)
            push!(new_parameter_names, variable_referred)
        end
    end

    # @show new_parameter_names
    for parameter_name in new_parameter_names
        push!(code_snippet.args[2].args[1].args, parameter_name)
        push!(code_snippet.args[1].args, parameter_name)
    end
    return esc(code_snippet)
end


function hei()
    println("hei")
end

# @for_each_cell_macro hei()

abstract type Equation end
abstract type NumericalFlux end
struct Burgers <: Equation end


struct Godunov{EquationType<:Equation} <: NumericalFlux
    eq::EquationType
end


(::Burgers)(u) = @SVector [0.5 * u .^ 2]
function (god::Godunov)(faceminus, faceplus)
    f(u) = 0.5 * u^2
    fluxminus = f(max.(faceminus, zero(faceminus)))
    fluxplus = f(min.(faceplus, zero(faceplus)))

    F = max.(fluxminus, fluxplus)
    return F
end
struct MyType
end


@kernel function for_each_cell_kernel(f, x, grid, y...)
    I = @index(Global)
    if I > 1 && I < size(x)[1]
        f(I - 1, I, I + 1, y...)
    end
end

function for_inner_each_cell(f, x, backend, grid, y...)
    #@show y
    ev = for_each_cell_kernel(backend, 64)(f, x, grid, y..., ndrange=size(x))
    #synchronize(backend)
end

function run_ka_test()
    N = 1000000
    Δx = 1 / N
    x = CUDA.cu(ones(N))
    y = CUDA.cu(ones(N))
    output = CUDA.cu(zeros(N))
    right = CUDA.cu(collect(1:N)) * Δx
    left = CUDA.cu(collect(0:N-1)) * Δx

    # x = ones(N)
    # y = ones(N)
    # output = zeros(N)
    # right = collect(1:N) * Δx
    # left = collect(0:N-1) * Δx
    F = Godunov(Burgers())

    function call_me_in_a_loop()
        @for_each_cell_macro for_inner_each_cell(x, get_backend(x), nothing) do ileft, imiddle, iright
            output[imiddle] -= 1 / Δx * (F(right[imiddle], left[iright]) - F(right[ileft], left[imiddle]))
            return nothing
        end
    end

    function loop_me()
        for _ in 1:10000
            call_me_in_a_loop()
        end
    end

    loop_me()
    @time loop_me()


end
if SinFVM.has_cuda_backend()
    run_ka_test()
end
#@show output
