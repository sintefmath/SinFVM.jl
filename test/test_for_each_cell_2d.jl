using SinSWE
using CUDA
using Test

for backend in get_available_backends()
    for gc in [1, 2]
        nx = 64
        ny = 32
        grid = SinSWE.CartesianGrid(nx, ny; gc=gc)
        

        
        output = SinSWE.convert_to_backend(backend, -42 * ones(Int64, nx + 2*gc, ny + 2*gc, 3))
        
        SinSWE.@fvmloop SinSWE.for_each_cell(backend, grid) do index
            output[index,1] = index[1]
            output[index,2] = index[2]
            output[index,3] = index[1] + index[2]
            
        end

        for y in 1:(ny+2*gc)
            for x in (1:nx+2*gc)
                CUDA.@allowscalar @test output[x, y,1] == x
                CUDA.@allowscalar @test output[x, y,2] == y
                CUDA.@allowscalar @test output[x, y,3] == x+y
            end
        end
    end
end