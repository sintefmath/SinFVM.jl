# What is SinFVM.jl?

SinFVM (SINTEF Finite Volume Methods) is a Julia framework for solving conservation and balance laws using the finite volume method, with a strong emphasis on modelling of surface water. It is developed and maintained by the
[Applied Computational Science](https://www.sintef.no/en/digital/departments-new/department-of-mathematics-and-cybernetics/research-group-applied-computational-science/)
research group at [SINTEF Digital](https://www.sintef.no/en/digital/).

If you are only interested in the modelling of surface water using SinFVM.jl, you can skip directly to TODO INSERT TUTORIAL REF HERE!

We consider a general (hyperbolic) conservation or balance law

$$u_t + \nabla_x \cdot F(u) = S(u)\qquad \text{on}\qquad D\times [0, T)$$

where $u:D\times[0,T)\to\mathbb{R}$ is a vector of conserved (balanced) quantities, $F$ is a flux function and $S$ represents source terms. It is common to discretize such equations with the finite volume method. 

