# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
Literate.markdown("examples/callbacks.jl", "docs/src/"; execute=false, preprocess=process_includes)

makedocs(modules = [SinFVM], 
    sitename="SinFVM.jl",
    draft=false,
    pages = [
        "Introduction" => "index.md",
        "Examples" => ["shallow_water_1d.md", "terrain.md", "optimization.md"],
        "Index" => "indexlist.md",
        "Public API" => "api.md"
    ])
