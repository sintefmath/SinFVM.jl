using CairoMakie

function terrain(x)
    wall_position = 100.0
    wall_height = 2.0
    if x < 100
        b = 2
    elseif 100 <= x <= 150
        b = (2 - 2 * sin(pi / 100 * (x - 100)))
    else
        b = 0.0
    end
    if true
        b += exp(-(x - 50)^2 / 100)
    end

    b += wall_height * exp(-(x - wall_position)^2 / 3.3)


    return b
end

x = LinRange(0, 200.0, 10_000)
f = Figure(size=(1600, 600), fontsize=24)
ax = Axis(
    f[1, 1],
)

lines!(ax, x, terrain.(x))
display(f)