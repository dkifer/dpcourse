module DP

using JuMP
using Gurobi

include("queries.jl")

""" Returns the closest nondecreasing vector to data """
function isotonic(data::Vector; nonneg=false)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    n = length(data)
    @variable(model, x[1:n])
    if nonneg
        @constraint(model, x .>= 0)
    end
    @constraint(model, x[2:end] .>= x[1:end-1])
    @objective(model, Min, sum((x - data).^2))
    optimize!(model)
    value.(x)
end

""" Solves a least square problem to fit the queries
assumes data is some array and datashape is its shape
the info variable is a vector of tuples of query, noisy query answer, and weight
to use in least squares optimization.
"""

function fit(datashape, info::Array{Tuple{Q, A, F}}; nonneg=false) where {Q <: Query, A <: Array, F <: Number}
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    numvars = reduce(*, datashape)
    @variable(model, x[1:numvars])
    mainvars = reshape(x, datashape)
    obj = 0
    for (query, noisyanswers, weight) in info
        numanswers = reduce(*, size(noisyanswers))
        tmpvars = @variable(model, [1:numanswers])
        newvars = reshape(tmpvars, size(noisyanswers))
        obj += sum((newvars - noisyanswers).^2)*weight
        @constraint(model, answer(query,mainvars) .== newvars)
    end
    @objective(model, Min, obj)
    optimize!(model)
    reshape(value.(x), datashape)
end


function mwem(queries, T, data, epsilon)
    datashape = size(data)
    synthdata = zeros(datashape...)
    sensitivity = maximum([sens(q) for q in queries])
    info = Nothing
    for i in 1:T
        epsilon_round = epsilon/T
        epsilon_select = epsilon_round/2
        epsilon_measure = epsilon_round/2
        errors = [sum(abs, answer(q, data) - answer(q, synthdata)) for q in queries]
        worst_query_id = noisy_max(errors, sensitivity=sensitivity, epsilon=epsilon_select)
        worst_query = queries[worst_query_id]
        worst_query_answer = protect(worst_query, data, epsilon_measure)
        if info == Nothing
            info = [(worst_query, worst_query_answer, 1.0)]
        else
            info = cat(info, (worst_query, worst_query_answer, 1.0), dims=1)
        end
        synthdata = fit(datashape, info, nonneg=true)
    end
    synthdata
end

end # end module
