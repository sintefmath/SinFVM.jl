
function get_variable_names_defined(expression)
    nothing
end



function get_variable_names_defined(expression::Expr)
    if expression.head == Symbol("=") && expression.args[1] isa Symbol
        return [expression.args[1]]
    else
        return get_variable_names_defined(expression.args)
    end
end

function get_variable_names_defined(expressions::Vector)
    all_symbols = []

    for expression in expressions
        symbols_from_expression = get_variable_names_defined(expression)

        if !isnothing(symbols_from_expression)
            all_symbols = vcat(all_symbols, symbols_from_expression)
        end
    end
    return all_symbols
end

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

function get_known_symbols()
    modules= [mod for mod in getfield.(Ref(Main),names(Main)) if typeof(mod)==Module && mod != Main]
    all_symbols = vcat((getfield.(Ref(m), names(m))  for m in modules)...)

    return all_symbols
end

macro fvmloop(code_snippet)
    all_symbols = get_known_symbols()
    
    parameter_names = get_variable_names_referred(code_snippet.args[2].args[1])
    function_body = code_snippet.args[2].args[2]

    all_variables_referenced = get_variable_names_referred(function_body.args)
    all_variables_referenced = Set(all_variables_referenced)
    
    all_variables_defined = get_variable_names_defined(function_body.args)
    
    for known_symbol in all_symbols
        delete!(all_variables_referenced, Symbol(known_symbol))
        delete!(all_variables_referenced, Symbol(".$known_symbol"))
    end

    for defined_symbol in all_variables_defined
        delete!(all_variables_referenced, defined_symbol)
        delete!(all_variables_referenced, Symbol(".$defined_symbol"))
    end
   
    new_parameter_names = []
    for variable_referred in all_variables_referenced
        if !(variable_referred in parameter_names)
            push!(new_parameter_names, variable_referred)
        end
    end

    for parameter_name in new_parameter_names
        push!(code_snippet.args[2].args[1].args, parameter_name)
        push!(code_snippet.args[1].args, parameter_name)
    end
    return esc(code_snippet)
end
