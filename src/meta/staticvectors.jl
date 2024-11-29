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

macro generate_static_vector_functions(max_dimension)
    expression_to_return = Expr[]
    for dimension in 1:max_dimension
        argument_to_vector_creation = [:(data[index, $i]) for i in 1:dimension]
        vector_creation = :(SVector{$(dimension),RealType}($(argument_to_vector_creation...)))
        function_definition = :(@inline extract_vector(::Val{$(dimension)}, data::AbstractArray{RealType}, index) where {RealType} = $(vector_creation))
        push!(expression_to_return, function_definition)

    end
    return esc(quote
        $(expression_to_return...)
    end)
end
@generate_static_vector_functions 10

@inline function set_vector!(::Val{n}, data, value, index) where {n}
    for i in 1:n
        data[index, i] = value[i]
    end
end
