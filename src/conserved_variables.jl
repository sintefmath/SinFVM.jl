import CUDA
import Adapt
import Meshes

struct ConservedVariables{RealVectorType}
    h::RealVectorType
    hu::RealVectorType
    hv::RealVectorType

    ConservedVariables(h, hu, hv) = new{typeof(h)}(h, hu, hv)
end

Base.size(cv::ConservedVariables) = size(cv.h)
function ConservedVariables(g::Meshes.CartesianGrid, ngc=2)
    (nx, ny) = size(g)
    shape = ((nx + 2 * ngc), (ny + 2 * ngc))

    return ConservedVariables(zeros(shape), zeros(shape), zeros(shape))
end

CUDA.cu(cv::ConservedVariables) = ConservedVariables(CUDA.cu(cv.h), CUDA.cu(cv.hu), CUDA.cu(cv.hv))

function Adapt.adapt_structure(to, cv::ConservedVariables)
    h = Adapt.adapt_structure(to, cv.h)
    hu = Adapt.adapt_structure(to, cv.hu)
    hv = Adapt.adapt_structure(to, cv.hv)

    return ConservedVariables(h, hu, hv)
end