using Documenter, Literate, SinFVM, Parameters

push!(LOAD_PATH, "../src/")
push!(LOAD_PATH, "../examples/")

examples_dir = realpath("examples/")
counterfile = 1


function process_includes(content)
    # Replace include statements with actual content
    while (m = match(r"include\(\"([^\"]+)\"\)", content)) !== nothing
        include_path = joinpath(examples_dir, m.captures[1])
        include_content = read(include_path, String)
        content = replace(content, m.match => include_content)
    end    
    return content
end
#Literate.markdown("examples/urban.jl", "docs/src/"; execute=false, preprocess=process_includes)
Literate.markdown("examples/terrain.jl", "docs/src/"; execute=false, preprocess=process_includes)
Literate.markdown("examples/optimization.jl", "docs/src/"; execute=false, preprocess=process_includes)
Literate.markdown("examples/shallow_water_1d.jl", "docs/src/"; execute=false, preprocess=process_includes)

makedocs(modules = [SinFVM], 
    sitename="SinFVM.jl",
    draft=false,
    pages = [
        "Introduction" => "index.md",
        "Examples" => ["shallow_water_1d.md", "terrain.md", "optimization.md"],
        "Index" => "indexlist.md",
        "Public API" => "api.md"
    ])