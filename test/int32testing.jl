function make_32bit(expression::Int64)
    return :(Int32($expression))
end

function make_32bit(expression::Float64)
    return :(Float32($expression))
end

function make_32bit(expression_we_do_not_care_about)
    expression_we_do_not_care_about
end

function make_32bit(expression::Expr)
    return Expr(make_32bit(expression.head), make_32bit(expression.args)...)
end

function make_32bit(expressions::Vector)
    return make_32bit.(expressions)
end

macro make_numeric_literals_32bits(definition)
    new_function = make_32bit(definition)
    esc(
        new_function
    )
end

@inline @make_numeric_literals_32bits function myotherfunc(a)
    return a + 2
end


@make_numeric_literals_32bits function my_function(a)
    b = myotherfunc(a)
    return b + a + 1+4.5+2.2f0
end

@show my_function(Int32(3))
