include("optimization.jl")  # Include the file containing the worstcase_l1 function

# Test case 1
z = [0.5, 0.2, 0.9, 0.1]
pbar = [0.1, 0.25, 0.4, 0.25]
xi = 0.5

p_opt, objective_value = worst_case_l1(z, pbar, xi)

println("Test case 1")
println("Optimal p: ", p_opt)
println("Objective value: ", objective_value)

z2 = [0.5, 0.2, 0.9, 0.1]   # Target distribution
pbar2 = [0.1, 0.25, 0.4, 0.25] # Initial probability distribution
w = [1.0, 1.0, 1.0, 1.0]    # Weights associated with each probability
xi2 = 0.5

result = worstcase_l1_w(z2, pbar2, w, xi2)
println("Test 2:")
println("For weights: ", w)
println("Updated Probability Distribution: ", result[1])
println("Objective Value: ", result[2])