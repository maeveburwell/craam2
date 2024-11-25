include("optimization.jl")  # Include the file containing the worstcase_l1 function

using Distributions

# Test case 1
z = [0.5, 0.2, 0.9, 0.1]
p̄ = [0.1, 0.25, 0.4, 0.25]
ξ = 0.5

p_opt, objective_value = worst_case_l1(z, p̄, ξ)
println("For z values: ", z)
println("For pbar values: ", p̄)
println("")
println("Test case 1 (worstcase_l1)")
println("Optimal p: ", p_opt)
println("Objective value: ", objective_value)
println()

# Test case 2

z2 = [0.5, 0.2, 0.9, 0.1]   # Target distribution
p̄2 = [0.1, 0.25, 0.4, 0.25] # Initial probability distribution
w = [1.0, 1.0, 1.0, 1.0]    # Weights associated with each probability
ξ2 = 0.5

result = worstcase_l1_w(z2, p̄2, w, ξ2)

@assert result[1] == p_opt #worst_case_l1 optimal values should be the same as worstcase_l1_w with uniform weights
@assert result[2] == objective_value #objectives should also be the same

println("Test case 2 (worstcase_l1_w):")
println("For uniform weights: ", w)
println("Updated Probability Distribution: ", result[1])
println("Objective Value: ", result[2])
println("")

# Test case 3

w2 = [1.0, 2.0, 3.0, 4.0]

result2 = worstcase_l1_w(z2, p̄2, w2, ξ2)

println("Test case 3 (worstcase_l1_w):")
println("For different weights: ", w2)
println("Updated Probability Distribution: ", result2[1])
println("Objective Value: ", result2[2])
println("")

# Test case 4

z3 = 2*rand(Dirichlet(5, 5))
p̄3 = rand(Dirichlet(5, 5))
w3 = 10*rand(Dirichlet(5, 5))

result3 = worstcase_l1_w(z3, p̄3, w3, ξ2)
println("Test case 4 (worstcase_l1_w):")
println("For pbar: ", p̄3)
println("For different weights: ", w3)
println("Updated Probability Distribution: ", result3[1])
println("Objective Value: ", result3[2])
println("")