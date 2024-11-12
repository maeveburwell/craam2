using LinearAlgebra
using Zygote
using DataStructures
using Printf
using Infinity


function gradients_func(p::Vector{Float64}, pr::Vector{Float64}, z::Vector{Float64},w::Vector{Float64})
    grad_que = []
    smallest = Infinity
    for iz in sorted_ind(z)
        assert(w[iz] > epsilon)
        if (w[iz] < smallest)
            pr.push(iz)
            smallest = w[iz]
        end
    end
    #case a: donor is less or equal to pbar
    #donor
    for i in 1:length(p)
        for j in pr
            if z[i] <= z[j] continue
            else 
                grad = (-z[i] + z[j]) / (w[i] + w[j])
                grad_que.push(grad < -epsilon ? derivative : 0)
                donors.push(i)
                receivers.push(j)
                donor_greater.push(true)
            end
        end
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
    grad_epsilon = 1e-5

    for k in 1:length(z)
        grad_que, steepest_grad = gradients_func(p, z, w)
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