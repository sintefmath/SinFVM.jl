using SinSWE
using CUDA
using Test

for backend in get_available_backends()
    nx = 10
    ny = 9
    grid = SinSWE.CartesianGrid(nx, ny)
    @test SinSWE.compute_dx(grid, XDIR) == 1.0 / nx
    @test SinSWE.compute_dx(grid, YDIR) == 1.0 / ny

    @test SinSWE.size(grid) == (nx + 2, ny + 2)
    @test SinSWE.inner_cells(grid, XDIR) == (nx, ny)
    @test SinSWE.inner_cells(grid, YDIR) == (nx, ny)
    @test SinSWE.inner_cells(grid, XDIR, 0) == (nx + 2, ny)
    @test SinSWE.inner_cells(grid, YDIR, 0) == (nx, ny + 2)

    @test SinSWE.middle_cell(grid, CartesianIndex(nx, ny), XDIR) == CartesianIndex(nx + 1, ny + 1)
    @test SinSWE.middle_cell(grid, CartesianIndex(nx, ny), YDIR) == CartesianIndex(nx + 1, ny + 1)
    for j in 1:ny
        for i in 1:nx
            @test SinSWE.middle_cell(grid, CartesianIndex(i, j), XDIR) == CartesianIndex(i + 1, j + 1)
            @test SinSWE.middle_cell(grid, CartesianIndex(i, j), YDIR) == CartesianIndex(i + 1, j + 1)

            @test SinSWE.left_cell(grid, CartesianIndex(i, j), XDIR) == CartesianIndex(i, j + 1)
            @test SinSWE.left_cell(grid, CartesianIndex(i, j), YDIR) == CartesianIndex(i + 1, j)

            @test SinSWE.right_cell(grid, CartesianIndex(i, j), XDIR) == CartesianIndex(i + 2, j + 1)
            @test SinSWE.right_cell(grid, CartesianIndex(i, j), YDIR) == CartesianIndex(i + 1, j + 2)
        end
    end

    # XDIR  
    output_left = SinSWE.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))
    output_middle = SinSWE.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))
    output_right = SinSWE.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))

    SinSWE.@fvmloop SinSWE.for_each_inner_cell(backend, grid, XDIR) do left, middle, right
        output_left[left, 1] = left[1]
        output_left[left, 2] = left[2]

        output_middle[middle, 1] = middle[1]
        output_middle[middle, 2] = middle[2]

        output_right[right, 1] = right[1]
        output_right[right, 2] = right[2]
    end

    for y in 1:(ny+2)
        for x in (1:nx+2)
            if x == 1 || y == 1 || x == nx + 2 || y == ny + 2
                CUDA.@allowscalar @test output_middle[x, y, 1] == -42
                CUDA.@allowscalar @test output_middle[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_middle[x, y, 1] == x
                CUDA.@allowscalar @test output_middle[x, y, 2] == y
            end

            if y == 1 || x == nx + 2 || x == nx + 1 || y == ny + 2
                CUDA.@allowscalar @test output_left[x, y, 1] == -42
                CUDA.@allowscalar @test output_left[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_left[x, y, 1] == x
                CUDA.@allowscalar @test output_left[x, y, 2] == y
            end
            if x == 1 || x == 2 || y == 1 || y == ny + 2
                CUDA.@allowscalar @test output_right[x, y, 1] == -42
                CUDA.@allowscalar @test output_right[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_right[x, y, 1] == x
                CUDA.@allowscalar @test output_right[x, y, 2] == y
            end
        end
    end


    # YDIR  
    output_left = SinSWE.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))
    output_middle = SinSWE.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))
    output_right = SinSWE.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))

    SinSWE.@fvmloop SinSWE.for_each_inner_cell(backend, grid, YDIR) do left, middle, right
        output_left[left, 1] = left[1]
        output_left[left, 2] = left[2]

        output_middle[middle, 1] = middle[1]
        output_middle[middle, 2] = middle[2]

        output_right[right, 1] = right[1]
        output_right[right, 2] = right[2]
    end

    for y in 1:(ny+2)
        for x in (1:nx+2)
            if x == 1 || y == 1 || x == nx + 2 || y == ny + 2
                CUDA.@allowscalar @test output_middle[x, y, 1] == -42
                CUDA.@allowscalar @test output_middle[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_middle[x, y, 1] == x
                CUDA.@allowscalar @test output_middle[x, y, 2] == y
            end

            if x == 1 || x == nx + 2 || y == ny + 1 || y == ny + 2
                CUDA.@allowscalar @test output_left[x, y, 1] == -42
                CUDA.@allowscalar @test output_left[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_left[x, y, 1] == x
                CUDA.@allowscalar @test output_left[x, y, 2] == y
            end
            if y == 1 || y == 2 || x == 1 || x == nx + 2
                CUDA.@allowscalar @test output_right[x, y, 1] == -42
                CUDA.@allowscalar @test output_right[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_right[x, y, 1] == x
                CUDA.@allowscalar @test output_right[x, y, 2] == y
            end
        end
    end
end