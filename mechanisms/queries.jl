include("mech.jl")

struct QueryException <: Exception
     msg::String
end

Base.showerror(io::IO, e::QueryException) = print(io, e.msg)

################### Query interface
abstract type Query end


# answers the query
answer(q::Query, data) = throw(QueryException("Not Implemented"))

# returns the query sensitivity
sens(q::Query) = throw(QueryException("Not IMplemented"))

# returns a privacy preserving answer
function protect(q::Query, data::Array, epsilon::Number)
    true_answer = answer(q, data)
    protected_answer = zeros(size(true_answer))
    for i in eachindex(true_answer)
        protected_answer[i] = laplace_mechanism(true_answer[i], sensitivity=sens(q), epsilon=epsilon)
    end
    protected_answer
end

#################### Marginal Queries

struct Marginal <: Query
    keepdims::Tuple
end

function answer(q::Marginal, data)
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

answer(q::RangeQuery, data) = [sum(data[q.left:q.right])]
sens(q::RangeQuery) = 1

################## Matrix Query

struct MatrixQuery <: Query
    mat::Matrix
end

answer(q::MatrixQuery, data) = q.mat * reshape(data, length(data))
sens(q::MatrixQuery) = maximum(sum(abs.(q.mat), dims=1))
function Base.show(io::IO, q::MatrixQuery)
    println(io, "MatrixQuery")
    show(io, "text/plain", q.mat)
end

##################

struct IDQuery <: Query
end

answer(q::IDQuery, data) = copy(data)
sens(q::IDQuery) = 1


##########################################
# Random range queries on 1-d dataset
#########################################
""" Returns a vector of `numqueries' random range queries on a 1-d dataset with d cells"""
function random_range_queries(numqueries, d)
    bounds = [sort([rand(1:d), rand(1:d)]) for _ in 1:numqueries]
    workload = [RangeQuery(x[1], x[2]) for x in bounds]
end

all_range_queries(d) = [RangeQuery(i,j) for i in 1:d for j in i:d]

###########################################
# Tree queries on 1-d dataset
##########################################

# tree
""" Returns a matrix query representing tree queries without the root """
function tree_queries(d, starting=1, ending=d; numsplits=2)
    mat = Vector{Int64}()
    splitsize = ceil(Int64, (ending-starting+1)/numsplits)
    for j in 1:numsplits
        left = (j-1) * splitsize + starting
        right = min(ending, left + splitsize - 1)
        tmp = zeros(Int64, 1, d)
        if left <= ending
            tmp[left:right] .= 1
            mat = cat(mat, tmp, dims=1)
            if right > left
                tmp2 = tree_queries(d, left, right, numsplits=numsplits)
                mat = cat(mat, tmp2.mat, dims=1)
            end
        end
    end
    MatrixQuery(mat)
end
