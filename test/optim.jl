using Optim, OptimTestProblems
problem = UnconstrainedProblems.examples["Rosenbrock"]

f = problem.f
initial_x = problem.initial_x

function very_slow(x)
    sleep(0.5)
    f(x)
end

start_time = time()
time_to_setup = zeros(1)
function advanced_time_control(x)
    @show fieldnames(typeof(x))
    println(" * Iteration:       ", x.iteration)
    so_far = time() - start_time
    println(" * Time so far:     ", so_far)
    if x.iteration == 0
        time_to_setup .= time() - start_time
    else
        expected_next_time = so_far + (time() - start_time - time_to_setup[1]) / (x.iteration)
        println(" * Next iteration ≈ ", expected_next_time)
        println()
        return expected_next_time < 13 ? false : true
    end
    println()
    false
end
optimize(very_slow, zeros(2), NelderMead(), Optim.Options(callback=advanced_time_control))
