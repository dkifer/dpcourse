include("fit.jl")
include("data.jl")


function example_iso1(epsi=0.5)
    data = [100, 1, 0, 0, 3, 5, 0, 99, 0, 0]
    sorted_data = sort(data)
    noisy_data = DP.laplace_mechanism(sorted_data, sensitivity=1, epsilon=epsi)
    postprocessed1 = sort(noisy_data)
    dpdata = DP.isotonic(noisy_data)
    println("Sorted: ", sorted_data)
    println("Noisy: ", round.(noisy_data, digits=3))
    println("Sorting the noisy data: ", round.(postprocessed1, digits=3))
    println("Isotonic regression: ", round.(dpdata, digits=3))
    # this is not very sorted now
    error_noisy_data = sum(x -> x^2, sorted_data - noisy_data)
    error_sorted = sum(x -> x^2, sorted_data - postprocessed1)
    error_isotonic = sum(x -> x^2, sorted_data - dpdata)
    println("Error of noisy data: ", error_noisy_data)
    println("Error of sorted noisy data: ", error_sorted)
    println("Error of isotonic: ", error_isotonic)

end

mean(x::Array) = sum(x)/length(x)

function example_iso2(epsi=0.5)
    data = getnet()
    sorted_data = sort(data)
    noisy_data = DP.laplace_mechanism(sorted_data, sensitivity=1, epsilon=epsi)
    dpdata = DP.isotonic(noisy_data)
    println("Noisy Data error: ", mean((sorted_data - noisy_data).^2))
    println("Postprocessed error: ", mean((sorted_data - dpdata).^2))
end


function example_marginal(epsi=3.0)
    datashape = (3,4,5)
    data = reshape(1:60, datashape)
    q1 = DP.Marginal((1,))
    q2 = DP.Marginal((2,))
    q3 = DP.Marginal((3,))

    answer1 = DP.answer(q1, data)
    answer2 = DP.answer(q2, data)
    answer3 = DP.answer(q3, data)

    dp1 = DP.laplace_mechanism(answer1, sensitivity=DP.sens(q1), epsilon=epsi/3)
    dp2 = DP.laplace_mechanism(answer2, sensitivity=DP.sens(q2), epsilon=epsi/3)
    dp3 = DP.laplace_mechanism(answer3, sensitivity=DP.sens(q3), epsilon=epsi/3)

    dphist = DP.fit(datashape, [(q1, dp1, 1.0), (q2, dp2, 1.0), (q3, dp3, 1.0)])

    for (i, q) in enumerate([q1, q2, q3])
        println("####################")
        println("Marginal ", i)
        println("####################")
        println("True:")
        show(DP.answer(q, data))
        println()
        println("DP:")
        show(round.(DP.answer(q, dphist), digits=3))
        println()
    end
end

sqerror(x::Number, y::Number) = (x-y)^2
sqerror(x::Array, y::Array) = sum(a -> a^2, x-y)

function evaluate_strategy(workload, strategy, d, data, epsilon)
    noisy_answers = DP.protect(strategy, data, epsilon)
    noisy_data = DP.fit(size(data), [(strategy, noisy_answers, 1.0)])
    mean([sqerror(DP.answer(q, data), DP.answer(q, noisy_data)) for q in workload])
end

function evaluate_mwem(workload, T, data, epsilon)
    noisy_data = DP.mwem(workload, T, data, epsilon)
    mean([sqerror(DP.answer(q, data), DP.answer(q, noisy_data)) for q in workload])
end

function example_range(epsilon=2, T=3, amount=100000)
    data = getfrank()[1:200]
    d = length(data)

    rangequeries = DP.all_range_queries(d)
    idquery = DP.IDQuery()
    error_id = mean([evaluate_strategy(rangequeries, idquery, d, data, epsilon) for _ in amount])

    tree = DP.tree_queries(d, numsplits=100)
    error_tree = mean([evaluate_strategy(rangequeries, tree, d, data, epsilon) for _ in amount])

    #error_mwem = mean([evaluate_mwem(rangequeries, T, data, epsilon) for _ in amount])

    println("#### Results: ####")
    println("ID Query mean squared error: ", error_id)
    println("Range Query mean squared error: ", error_tree)
    #println("MWEM mean squared error: ", error_mwem)
end
