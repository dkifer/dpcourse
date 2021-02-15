import Distributions

function laplace_mechanism(;result::Number, sensitivity::Number, epsilon::Number)
    @assert sensitivity > 0
    @assert epsilon > 0
    lap = Distributions.Laplace(0.0, sensitivity/epsilon) # mean 0, scale=sensitivity/epsilon
    result + rand(lap)
end


function geometric_mechanism(; result::Int, sensitivity::Int, epsilon::Number)
    @assert sensitivity > 0
    @assert epsilon > 0
    p = 1-exp(-epsilon/sensitivity)
    geo = Distributions.Geometric(p) 
    result + rand(geo) - rand(geo)
end

function noisy_max(;answers::Vector{S}, sensitivity::Number, epsilon::Number) where S <: Number
    noisy_answers = [laplace_mechanism(result=x, sensitivity=2*sensitivity, epsilon=epsilon) for x in answers]
    return argmax(noisy_answers)
end

function sparse_vector(;T::Number, N::Int, answers::Vector{S}, sensitivity::Number, epsilon::Number) where S <: Number
    noisy_T = laplace_mechanism(;result=T, sensitivity=sensitivity, epsilon=epsilon/2)
    answered = 0
    i=1
    output = Bool[] # empty array of booleans
    while(answered < N && i <= length(answers))
        noisy_query = laplace_mechanism(result = answers[i], sensitivity=2*N, epsilon=epsilon/2)
	i = i + 1
	if noisy_query >= noisy_T
	   append!(output, true)
	   answered = answered + 1
	else
           append!(output, false)
	end
    end
    output
end
