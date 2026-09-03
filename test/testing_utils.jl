# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Shared helpers for the test suite.
#
# Deliberately not named `test_*.jl`: `runtests.jl` auto-includes every `test/test_*.jl`,
# and this file holds no tests of its own.

using VolumeFluxes

"""
    SHOW_PLOTS

Whether the test suite should render figures. Off by default.

Several test files draw diagnostic figures, and a few do it unconditionally in the middle
of the simulation they are testing. Rendering costs about 0.3 s per figure (roughly five
times what building it costs) and produces output nobody looks at during a test run, so it
is skipped unless explicitly asked for:

    VOLUMEFLUXES_TEST_PLOTS=true julia --project -t auto test/runtests.jl
"""
const SHOW_PLOTS = get(ENV, "VOLUMEFLUXES_TEST_PLOTS", "false") == "true"

"""
    maybe_display(figure)

`display(figure)` when [`SHOW_PLOTS`](@ref) is set, otherwise a no-op.
"""
maybe_display(figure) = SHOW_PLOTS ? display(figure) : nothing
