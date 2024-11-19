include("optimization.jl")  # Include the file containing the worstcase_l1 function

# Test case 1
z = [0.5, 0.2, 0.9, 0.1]
pbar = [0.1, 0.25, 0.4, 0.25]
xi = 0.5

p_opt, objective_value = worst_case_l1(z, pbar, xi)
println("For z values: ", z)
println("For pbar values: ", pbar)
println("")
println("Test case 1 (worst_case_l1)")
println("Optimal p: ", p_opt)
println("Objective value: ", objective_value)
println()
z2 = [0.5, 0.2, 0.9, 0.1]   # Target distribution
pbar2 = [0.1, 0.25, 0.4, 0.25] # Initial probability distribution
w = [1.0, 1.0, 1.0, 1.0]    # Weights associated with each probability
xi2 = 0.5

result = worstcase_l1_w(z2, pbar2, w, xi2)

@assert result[1] == p_opt #worst_case_l1 should be the same as worstcase_l1_w with uniform weights

println("Test case 2 (worstcase_l1_w):")
println("For uniform weights: ", w)
println("Updated Probability Distribution: ", result[1])
println("Objective Value: ", result[2])
println("")

w2 = [1.0, 2.0, 3.0, 4.0]

result2 = worstcase_l1_w(z2, pbar2, w2, xi2)
println("Test case 3 (worstcase_l1_w):")
println("For different weights: ", w2)
println("Updated Probability Distribution: ", result2[1])
println("Objective Value: ", result2[2])
println("")