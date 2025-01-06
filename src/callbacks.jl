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

using Parameters

"""
	IntervalWriter{WriterType}

A structure for writing data at specified intervals using a user
specified writer of type `WriterType`. This avoids having every
timestep written.

# Type Parameters

- `WriterType`: The type of the writer used for data output.

# Description

`IntervalWriter` is designed to periodically write data during simulations.
It utilizes a writer of the specified `WriterType` to output data at defined 
intervals.
"""
@with_kw mutable struct IntervalWriter{WriterType}
	current_t::Float64 = 0.0
	step::Float64
	writer::WriterType
end

IntervalWriter(step::Real, writer::Base.Callable) = IntervalWriter(step=step, writer=writer)
IntervalWriter(writer::Base.Callable) = IntervalWriter(step=1.0, writer=writer)

function (writer::IntervalWriter)(t, simulator)
	dt = SinFVM.current_timestep(simulator)
	if t + dt >= writer.current_t
		writer.writer(t, simulator)
		writer.current_t += writer.step
	end
end

"""
    MultipleCallbacks(callbacks)

A callback aggregator that allows multiple callbacks to be executed sequentially.

# Arguments
- `callbacks`: Collection of callback functions to be executed

# Fields
- `callbacks`: Stored collection of callback functions

Each callback in the collection should be a function that accepts two arguments:
- `t`: The current time
- `simulator`: The simulator state/object

# Example
"""
struct MultipleCallbacks
	callbacks::Any
end

function (mc::MultipleCallbacks)(t, simulator)
	for c in mc.callbacks
		c(t, simulator)
	end
end
