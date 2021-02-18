include("mech.jl")

struct QueryException <: Exception
     msg::String
end

Base.showerror(io::IO, e::QueryException) = print(io, e.msg)

################### Query interface
abstract type Query end


# answers the query
answer(q::Query, data::Array) = throw(QueryException("Not Implemented"))

# returns the query sensitivity
sens(q::Query) = throw(QueryException("Not IMplemented"))

# returns a privacy preserving answer
function protect(q::Query, data::Array, epsilon::Number) 
    true_answer = answer(q, data)
    protected_answer = zeros(size(true_answer))
    for i in eachindex(true_answer)
        protected_answer[i] = laplace_mechanism(result=true_answer[i], sensitivity=sens(q), epsilon=epsilon)
    end
    protected_answer
end

#################### Marginal Queries

struct Marginal <: Query
    keepdims::Tuple
end

function answer(q::Marginal, data::Array)
    numdims = length(size(data))
    sumdims = [i for i in 1:numdims if !(i in q.keepdims)]
    sum(data, dims=sumdims)
end

sens(q::Marginal) = 1

################## Range Queries

struct RangeQuery <: Query
    left::Int
    right::Int
end

answer(q::RangeQuery, data::Array) = sum(data[q.left:q.right])
sens(q::RangeQuery) = 1

