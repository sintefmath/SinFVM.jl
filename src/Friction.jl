include("SwimTypeMacros.jl")

## -------------------------------
## FRICTION FUNCTIONS
## -------------------------------

@inline @make_numeric_literals_32bits function 
    friction_bsa2012(c, h, u, v)

    velocity = sqrt(u*u + v*v)
    denom = cbrt(h)*h
    return -c*velocity/denom
end

@inline @make_numeric_literals_32bits function 
    friction_fcg2016(c, h, u, v)
    
    velocity = sqrt(u*u + v*v)
    denom = cbrt(h)*h*h
    return -c*velocity/denom
end

@inline @make_numeric_literals_32bits function 
    friction_bh2021(c, h, u, v)
    
    velocity = sqrt(u*u + v*v)
    denom = h*h
    return -c*velocity/denom
end

