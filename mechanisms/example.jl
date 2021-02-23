include("fit.jl")
include("data.jl")


function example_iso1(epsi=0.5)
    data = [100, 1, 0, 0, 3, 5, 0, 99, 0, 0]
    sorted_data = sort(data)
    noisy_data = DP.laplace_mechanism(sorted_data, sensitivity=1, epsilon=epsi)
#    dpdata = DP.isotonic(noisy_data)
    println("Sorted: ", sorted_data)
    println("Noisy: ", round.(noisy_data, digits=3))
#    println("Postprocessed: ", round.(dpdata, digits=3))
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

    tmphist = DP.fit(datashape, [(q1, dp1, 1.0), (q2, dp2, 1.0), (q3, dp3, 1.0)])
    dphist = reshape(tmphist, datashape)

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
