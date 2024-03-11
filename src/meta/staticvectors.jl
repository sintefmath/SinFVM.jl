macro generate_static_vector_functions(max_dimension)
    expression_to_return = Expr[]
    for dimension in 1:max_dimension
        argument_to_vector_creation = [:(data[index, $i]) for i in 1:dimension]
        vector_creation = :(SVector{$(dimension),RealType}($(argument_to_vector_creation...)))
        function_definition = :(extract_vector(::Val{$(dimension)}, data::AbstractMatrix{RealType}, index) where {RealType} = $(vector_creation))
        push!(expression_to_return, function_definition)

    end
    return esc(quote
        $(expression_to_return...)
    end)
end
@generate_static_vector_functions 10

function set_vector!(::Val{n}, data, value, index) where {n}
    for i in 1:n
        data[index, i] = value[i]
    end
end
