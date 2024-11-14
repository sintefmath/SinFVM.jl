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
