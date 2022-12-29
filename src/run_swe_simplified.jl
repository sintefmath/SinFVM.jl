function run_swe_simplified(terrain::Matrix{<:Real},
                            final_time,
                            rain_function::Union{Function, Matrix{<:Real}};
                            savedir = "data",
                            rstep = 100, # reportstep length
                            friction_function = friction_fcg2016,
                            infiltration_function = infiltration_horton_fcg,
                            friction_constant = 0.03^2,
                            theta = 1.3)
                            
