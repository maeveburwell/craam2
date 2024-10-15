include("optimization.jl")  # Include the file containing the worstcase_l1 function

# Test case 1
z = [0.5, 0.2, 0.9, 0.1]
pbar = [0.25, 0.25, 0.25, 0.25]
xi = 0.5

# Run the function
p_opt, objective_value = worst_case_l1(z, pbar, xi)

# Display the results
println("Test case 1")
println("Optimal p: ", p_opt)
println("Objective value: ", objective_value)

# Add more test cases as needed
