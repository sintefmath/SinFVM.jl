
function get_variable_names_defined(expression)
    nothing
end


function get_variable_names_defined(expression::Expr)
    if expression.head == Symbol("=") && expression.args[1] isa Symbol
        # a = b
        return [expression.args[1]]
    elseif expression.head == :tuple && expression.args[1] isa Symbol && expression.args[end] isa Expr && expression.args[end].head == :(=)
        # a, b, ... = expr
        defined_names = Symbol[]

        for s in expression.args[2:end-1]
            if s isa Symbol
                push!(defined_names, s)
            end
        end

        push!(defined_names, expression.args[end].args[1])

        return defined_names
    elseif expression.head == :(=) && expression.args[1] isa Expr && expression.args[1].head == :(tuple) && any([x isa Symbol for x in expression.args[1].args])
        # a,b,c,.. = f()   
        defined_names = Symbol[]     
        for s in expression.args[1].args
            if s isa Symbol # We support expressions where only parts are symbols
                push!(defined_names, s)
            end
        end

        return defined_names
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

    # Delete functions. This could be an issue down the line...
    # TODO: Review this. Currently we need it for the minmod function in LinearReconstruction for some reason...
    all_functions = []
    for variable in all_variables_referenced
        if isdefined(SinSWE, variable) &&  isa(getfield(SinSWE, variable), Function)
            push!(all_functions, variable)
        end
    end
    for function_name in all_functions
        delete!(all_variables_referenced, function_name)
    end
   
    # TODO: Delete all module names
    # TODO: Use current module name insteadof referencing SinSWE
    delete!(all_variables_referenced, :SinSWE)
    
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
