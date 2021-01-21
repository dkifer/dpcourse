using JuMP
using Gurobi

""" generate an array or matrix full of 1 or -1 independently with prob p 
size is either an integer (if we want to generate an array) or a tuple (if we want
a matrix)
"""
function pm_one(size, p::Float64)
    bernoulli_array = rand(size...) .< p  # pointwise compare random numbers to p
    2 .* bernoulli_array .- 1
end


""" generate a dataset of n people, with approximately pn of them having +1 and the rest having -1 """
function make_data(n::Int, p::Float64)
    pm_one(n, p)
end


""" generate num_queries random queries whose coefficients are +1 or -1
Compute noisy answers for them by adding uniform noise, where the noise can be at most bound.
"""
function noisy_queries(num_queries::Int, data::Array, bound::Float64)
    n = length(data)
    size = (num_queries, n)
    query_matrix = pm_one(size, 0.5)
    answers = query_matrix * data
    noisy_answers = answers + bound .* (2 .* rand(num_queries) .- 1)
    (query_matrix, noisy_answers)
end


"""
Attempt to reconstruct the database using the noisy queries
"""
function attack_db(query_matrix, noisy_answers, bound)
    model = Model(Gurobi.Optimizer)
    num_queries, n = size(query_matrix)
    @variable(model, -1 <= x[1:n] <= 1)
    @constraint(model, query_matrix * x .<= noisy_answers .+ bound)
    @constraint(model, noisy_answers .- bound .<= query_matrix * x)
    @objective(model, Min, 1)
    optimize!(model)
    2 .* (value.(x) .>= 0) .- 1
end


"""  When you call this function, you must name the parameters you want to provide
e.g., example(n=100)
"""
function example(;n=100, bound=5.0, num_queries=200)
    data = make_data(n, 0.5)
    queries, answers = noisy_queries(num_queries, data, bound)
    x = attack_db(queries, answers, bound)
    num_correct = sum(x .== data)
    println("\n*****\n Got $(num_correct) correct out of $(n)\n****")
    queries, answers, data, x
end
