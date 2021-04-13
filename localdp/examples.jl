using LinearAlgebra

const maxval = 4
const numbits = 2
const data = [1, 1, 2, 3, 4]
const mhash = 4

""" holds all reports of clients. Each element is an array """
mutable struct Report{T}
   thedata
   encoded
   perturbed
   reconstructed  # unary encoding of reconstruction
   Report{T}() where {T} = new(data, missing, missing, missing)
   Report{T}(dataparam) where {T} = new(dataparam, missing, missing, missing)
end

encode(rs::Array{Report{T}}, epsilon) where {T} = [encode(r, epsilon) for r in rs]
perturb(rs::Array{Report{T}}, epsilon) where {T} = [perturb(r, epsilon) for r in rs]
reconstruct(rs::Array{Report{T}}, epsilon) where {T} = [reconstruct(r, epsilon) for r in rs]


function Base.show(io::IO, r::Report{T}) where {T}
    println(io, "Report{",T,"}")
    print(io, "Data: ")
    show(io, r.thedata)
    println(io)
    println(io, "Encoded: ")
    show(io, "text/plain", r.encoded)
    println(io)
    println(io, "Perturbed: ")
    show(io, "text/plain", r.perturbed)
    println(io)
    println(io, "Reconstructed: ")
    show(io, "text/plain", r.reconstructed)

end

#############################################################################
""" For reports that use unary encoding, RR on each bit"""
abstract type Unary end
UnaryReport = Report{Unary}

function encode(r::UnaryReport, epsilon)
   r.encoded = map(r.thedata) do d
      x = zeros(maxval) .- 1
      x[d] = 1
      x
   end
   r
end

function perturb(r::UnaryReport, epsilon)
   r.perturbed = map(r.encoded) do d
      [randomized_response(x, epsilon/2) for x in d]
   end
   r
end

function reconstruct(r::UnaryReport, epsilon)
    r.reconstructed = [(p .+ 1) ./ 2  for p in r.perturbed] 
    r
end

#############################################################################
""" For reports that use the value representation, joint RR"""
abstract type RRUnary end
RRUnaryReport = Report{RRUnary}


function encode(r::RRUnaryReport, epsilon)
   r.encoded = r.thedata
   r
end

function perturb(r::RRUnaryReport, epsilon)
   r.perturbed = map(r.encoded) do d
      randomized_multi_response(d, maxval, epsilon) 
   end
   r
end

function reconstruct(r::RRUnaryReport, epsilon)
   rrunary_matrx = [ exp(epsilon) 1 1 1;
      		    1  exp(epsilon) 1 1;
		    1 1  exp(epsilon) 1;
		    1 1 1  exp(epsilon)] ./ (maxval - 1 + exp(epsilon))
   r.reconstructed = map(r.perturbed) do d
       indicator = zeros(maxval) 
       indicator[d] = 1
       rrunary_matrx \ indicator
   end
   r
end

#############################################################################
""" For reports that use binary encoding, RR on each bit
first bit is 1 if (d-1) & 1 == 1, second bit is 1 if (d-1) & 2 == 1
"""
abstract type Binary end
BinaryReport = Report{Binary}

function encode(r::BinaryReport, epsilon)
   r.encoded = map(r.thedata) do d
      x = zeros(numbits)
      x[1] = if (d == 2 || d == 4) 1 else -1 end
      x[2] = if (d == 3 || d == 4) 1 else -1 end
      x
   end
   r
end


function perturb(r::BinaryReport, epsilon)
   r.perturbed = map(r.encoded) do d
      [randomized_response(x, epsilon/2) for x in d]
   end
   r
end

function reconstruct(r::BinaryReport, epsilon)
   r.reconstructed = map(r.perturbed) do d
      indicator = zeros(maxval)
      pos = 1 + (d[1] > 0 ? 1 : 0) + 2 * (d[2] > 0 ? 1 : 0)
      indicator[pos] = 1
      p = exp(epsilon/2)/(1+exp(epsilon/2))
      thematrix = [p^2      p*(1-p)  p*(1-p)  (1-p)^2; 
	           p*(1-p)  p^2      (1-p)^2  p*(1-p); 
		   p*(1-p)  (1-p)^2  p^2      p*(1-p); 
		   (1-p)^2  p*(1-p)  p*(1-p)  p^2]
      thematrix \ indicator
   end
   r
end

#############################################################################
""" For reports that choose a bit at random  Reports are tuples. First element is bit position, second is value"""
abstract type RBit end
RBitReport = Report{RBit}

rbit_matrix = [-1  1 -1 1; 
	       -1 -1  1 1]

function encode(r::RBitReport, epsilon)
   r.encoded = map(r.thedata) do d
      chosen_bit = rand(1:size(rbit_matrix, 1))
      valuebit = rbit_matrix[chosen_bit, d]
      (chosen_bit, valuebit)
   end
   r
end

function perturb(r::RBitReport, epsilon)
   r.perturbed = map(r.encoded) do d
      pert = randomized_response(d[2], epsilon)
      (d[1], pert)
   end
   r
end

function reconstruct(r::RBitReport, epsilon)
   r.reconstructed = map(r.perturbed) do d
       rbit_matrix[d[1], :]  * d[2]
   end
   r
end

#############################################################################
""" For reports that use a random yes/no question. First element is the indicator
second is the answer """ 
abstract type YNBit end
YNBitReport = Report{YNBit}


function encode(r::YNBitReport, epsilon)
   yn_matrix = [rand([-1,1]) for _ in 1:mhash, _ in 1:maxval]
   r.encoded = map(r.thedata) do d
      chosen_bit = rand(1:size(yn_matrix, 1))
      valuebit = yn_matrix[chosen_bit, d]
      (copy(yn_matrix[chosen_bit,:]), valuebit)
   end
   r
end

function perturb(r::YNBitReport, epsilon)
   r.perturbed = map(r.encoded) do d
      pert = randomized_response(d[2], epsilon)
      (d[1], pert)
   end
   r
end

function reconstruct(r::YNBitReport, epsilon)
   r.reconstructed = map(r.perturbed) do d
       d[1] * d[2]
   end
   r
end

#############################################################################
""" For reports that use a balanced y/n questions where matrix of set indicators has 0 column sum
returns tuple where first element is row number and second is answer to that question
"""
abstract type Balanced end
BalancedReport = Report{Balanced}

balanced_matrix = [ 1  1  1   1; 
	            1  1  1  -1; 
	            1  1 -1   1;
	            1  1 -1  -1;
                    1 -1  1   1; 
	            1 -1  1  -1; 
	            1 -1 -1   1;
	            1 -1 -1  -1;
                   -1  1  1   1; 
	           -1  1  1  -1; 
	           -1  1 -1   1;
	           -1  1 -1  -1;
                   -1 -1  1   1; 
	           -1 -1  1  -1; 
	           -1 -1 -1   1;
	           -1 -1 -1  -1
	    ]

function encode(r::BalancedReport, epsilon)
   r.encoded = map(r.thedata) do d
      chosen_bit = rand(1:size(balanced_matrix, 1))
      valuebit = balanced_matrix[chosen_bit, d]
      (chosen_bit, valuebit)
   end
   r
end

function perturb(r::BalancedReport, epsilon)
   r.perturbed = map(r.encoded) do d
      pert = randomized_response(d[2], epsilon)
      (d[1], pert)
   end
   r
end

function reconstruct(r::BalancedReport, epsilon)
   r.reconstructed = map(r.perturbed) do d
       balanced_matrix[d[1], :]  * d[2]
   end
   r
end

#############################################################################



""" applies differetially private randomized response to the +/- 1 bit. The output expected value is the bit"""
function randomized_response(thebit, epsilon) 
   p = exp(epsilon)/(1 + exp(epsilon))
   multiplier = (exp(epsilon) + 1) / (exp(epsilon) - 1)
   multiplier * (if rand() <= p  1   else   -1   end) * thebit
end


function randomized_multi_response(theval, numchoices, epsilon)
   p = (exp(epsilon)-1)/(numchoices - 1 + exp(epsilon))
   result = if rand() <= p
      theval
   else
      rand(1:numchoices)
   end
   result
end


#########################################
# For testing
########################################

function test1()
   x = [randomized_response(-1, log(2)) for _ in 1:1000000]
   y = [randomized_response(1, log(2)) for _ in 1:1000000]
   println(count(s -> s==3, x)/length(x), " should be about 1/3")
   println(count(s -> s==3, y)/length(y), " should be about 2/3")
   println(sum(x)/length(x), " should be about -1")
   println(sum(y)/length(y), " should be about 1")
end

function test2()
   println("data is ", data)
   runary = UnaryReport()
   r_rrunary = RRUnaryReport()
   rbinary = BinaryReport()
   rbit = RBitReport()
   ryn = YNBitReport()
   rbal = BalancedReport()
   println("###########################")
   println("Unary")
   display(encode(runary, 1).encoded)
   println("Perturbed")
   display(perturb(runary, 1).perturbed)
   println("###########################")
   println("RRUnary")
   display(encode(r_rrunary, 1).encoded)
   println("Perturbed")
   display(perturb(r_rrunary, 1).perturbed)
   println("###########################")
   println("Binary")
   display(encode(rbinary, 1).encoded)
   println("Perturbed")
   display(perturb(rbinary, 1).perturbed)
   println("###########################")
   println("Rbit")
   display(encode(rbit, 1).encoded)
   println("Perturbed")
   display(perturb(rbit, 1).perturbed)
   println("###########################")
   println("YN")
   display(encode(ryn, 1).encoded)
   println("Perturbed")
   display(perturb(ryn, 1).perturbed)
   println("###########################")
   println("Balanced")
   display(encode(rbal, 1).encoded)
   println("Perturbed")
   display(perturb(rbal, 1).perturbed)
   println("###########################")
end

##################################################
#### Evaluation
##################################################

function evaluator(;epsilon=5, repetitions=100000, mydata=data)
    mymethods = [Unary, RRUnary, Binary, YNBit, Balanced]
    encoded_data = zeros(maxval)
    for d in mydata
       encoded_data[d] += 1
    end
    for m in mymethods
        allreports = [reconstruct(perturb(encode(Report{m}(mydata), epsilon), epsilon), epsilon) for _ in 1:repetitions]
	thecounts = [sum(r.reconstructed) for r in allreports]
	println("Method: ", m)
	println("Data")
	show(encoded_data)
	println()
	println("Mean")
	show(sum(thecounts)/length(thecounts))
	println()
	println("Variance")
	show(sum([(tc - encoded_data).^2 for tc in thecounts])/length(thecounts))
	println()
	println()
	println("----------")
	println()
    end
end
