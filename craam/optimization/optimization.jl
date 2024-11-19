using LinearAlgebra
using Zygote
using DataStructures
using Printf

function worst_case_l1(z::Vector{Float64}, pbar::Vector{Float64}, xi::Float64)
    @assert maximum(pbar) <= 1 + 1e-9 && minimum(pbar) >= -1e-9 "values must be between 0 and 1"
    @assert xi >= 0 "xi must be nonnegative"
    @assert length(z) > 0 && length(z) == length(pbar) "z's values needs to be same length as pbar's values"
    @assert sum(pbar) == 1 "Values of pbar must sum to one"

    xi = clamp(xi, 0, 2)
    size = length(z) #size = 4
    sorted_ind = sortperm(z) #creates a list of the indexes of z sorted by the corresponding values in ascending order
                             #e.g. z = [0.1, 0.6, 0.2, 0.5] -> sorted_ind = [1, 4, 2, 3]

    out = copy(pbar) #duplicate it
    k = sorted_ind[1] #index begins at 1: e.g. from before, k = 1

    epsilon = min(xi / 2, 1 - pbar[k]) #1st iteration: min(0.5/2, 1 - 0.25), though tbf, pbar[k] for all k is going to be 0.25
    out[k] += epsilon #out[k] for the first iteration would return 0.5 (out[1])
    i = size

    while epsilon > 0 && i > 0
        k = sorted_ind[i] #second iteration:  k = sorted_ind[3] = 2 #third iteration: k = sorted[2] = 4
        difference = min(epsilon, out[k]) #min(0.25, out[2]) = min(0.25,0.25) = 0.25 # third iteration: min(0, 0.25) = 0
        out[k] -= difference #out[2] - 0.25 = 0.25 - 0.25 = 0 #third iteration: out[4] = out[4] - 0 = 0.25
        epsilon -= difference #0 -> 0
        i -= 1
    end

    return out, dot(out, z)

end

function gradients_func(p::Vector{Float64}, z::Vector{Float64})
    grad = Zygote.gradient((p) -> sum((p .- z).^), p)
    return grad[1]
end

function determine_receiver(p::Vector{Float64}, donor::Float64, gradients::Vector{Float64})
    receiver = argmax(gradients .* (1 .- (1:length(p) .== donor)))
    return gradients[receiver] > gradients[donor] ? receiver : 0
end

function worstcase_l1_w(z::Vector{Float64}, pbar::Vector{Float64}, w::Vector{Float64}, xi::Float64)
    @assert maximum(pbar) <= 1 + 1e-9 && minimum(pbar) >= -1e-9 "values must be between 0 and 1"
    @assert xi >= 0 "xi must be nonnegative"
    @assert length(z) > 0 && length(z) == length(pbar) "z's values needs to be same length as pbar's values"
    @assert sum(pbar) == 1 "Values of pbar must sum to one"

    epsilon = 1e-10
    p = copy(pbar)
    xi_rest = xi
    grad_que = []
    grad_epsilon = 1e-5

    for k in 1:length(z)
        steepest_grad = gradients_func(p, z)
        push!(grad_que, (steepest_grad[k], k))

        while length(grad_que) > 1 && grad_que[1][1] < grad_que[end][1] - grad_epsilon
            popfirst!(grad_que)
        end

        for grad in grad_que

            weight, donor = grad
            receiver = determine_receiver(p, donor, steepest_grad)

            if receiver == 0 continue end
            
            donor_greater = (steepest_grad[donor] > steepest_grad[receiver])

            if donor_greater && p[donor] <= pbar[donor] + epsilon continue end

            if !donor_greater && p[donor] > pbar[donor] + epsilon continue end

            if p[donor] < epsilon continue end

            weight_change = donor_greater ? (-w[donor] + w[receiver]) : (w[donor] + w[receiver])
            @assert weight_change > 0

            donor_step = min(xi_rest / weight_change, 
                             p[donor] > pbar[donor] + epsilon ? (p[donor] - pbar[donor]) : p[donor])
            p[donor] -= donor_step
            p[receiver] += donor_step
            xi_rest -= donor_step * weight_change

            if xi_rest < epsilon break end
        end
        if xi_rest < epsilon break end
    end

    objective = dot(p, z)
    return (p, objective)
end