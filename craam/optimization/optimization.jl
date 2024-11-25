using LinearAlgebra
using DataStructures
using Infinity

struct GradientsL1_w
    grads :: Vector{Float64}
    donors :: Vector{Int}
    receivers::Vector{Int}
    donor_greater::Vector{Bool}
    sorted::Vector{Float64}
end

function GradientsL1_w(z::Vector{Float64}, w::Vector{Float64})
    ϵ = 1e-8
    element_count = Int(length(z))

    @assert length(w) == element_count

    grads = Float64[]
    donors = Int[]
    receivers = Int[]
    donor_greater = Bool[]

    # Identifing possible receivers
    z_increasing = sortperm(z)
    possible_receivers = Int[]
    smallest_w = Inf

    for iz in z_increasing
        @assert w[iz] > ϵ
        if w[iz] < smallest_w
            push!(possible_receivers, iz)
            smallest_w = w[iz]
        end
    end

    # Computing grads for donor-receiver pairs
    for i = 1:element_count
        for j in possible_receivers
            if z[i] <= z[j] continue end
            # Case a: donor ≤ p̄ value
            grad = (-z[i] + z[j]) / (w[i] + w[j])
            push!(grads, grad < -ϵ ? grad : 0)
            push!(donors, i)
            push!(receivers, j)
            push!(donor_greater, false)
        end
    end

    # Case b: donor > p̄ value
    for i in possible_receivers
        for j in possible_receivers
            if z[i] <= z[j] continue end
            if abs(w[i] - w[j]) > ϵ && w[i] < w[j]
                grad = (-z[i] + z[j]) / (-w[i] + w[j])
                push!(grads, grad < -ϵ ? grad : 0)
                push!(donors, i)
                push!(receivers, j)
                push!(donor_greater, true)
            end
        end
    end

    sorted = sortperm(grads)

    return GradientsL1_w(grads, donors, receivers, donor_greater, sorted)
end

function steepest_solution(gradients::GradientsL1_w, index::Int)
    @assert index >= 1 && index <= Int(length(gradients.sorted))
    e = Int(gradients.sorted[index])
    return gradients.grads[e], gradients.donors[e], gradients.receivers[e], gradients.donor_greater[e]
end

function worstcase_l1_w(z::Vector{Float64}, p̄::Vector{Float64}, w::Vector{Float64}, ξ::Float64)
    @assert maximum(p̄) <= 1 + 1e-9 && minimum(p̄) >= -1e-9 "values must be between 0 and 1"
    @assert ξ >= 0 "ξ must be nonnegative"
    @assert length(z) > 0 && length(z) == length(p̄) "z's values needs to be same length as p̄'s values"
    @assert isapprox(sum(p̄), 1, atol=1e-5) "Values of p̄ must sum to one"

    ϵ = 1e-10
    xi_rest = ξ
    grad_epsilon = 1e-5
    grad_que = [] #tuple

    gradients = GradientsL1_w(z, w)

    for k in 1:length(z)
        push!(grad_que, steepest_solution(gradients, Int(k)))

        while length(grad_que) > 1 && grad_que[1][1] < grad_que[end][1] - grad_epsilon
            popfirst!(grad_que)
        end

        for g in grad_que

            _, donor, receiver, donor_greater = g

            if receiver == 0 continue end

            if donor_greater && p̄[donor] <= p̄[donor] + ϵ continue end

            if !donor_greater && p̄[donor] > p̄[donor] + ϵ continue end

            if p̄[donor] < ϵ continue end

            weight_change = donor_greater ? (-w[donor] + w[receiver]) : (w[donor] + w[receiver])
            @assert weight_change > 0

            donor_step = min(xi_rest / weight_change, 
                             p̄[donor] > p̄[donor] + ϵ ? (p̄[donor] - p̄[donor]) : p̄[donor])
            p̄[donor] -= donor_step
            p̄[receiver] += donor_step
            xi_rest -= donor_step * weight_change

            if xi_rest < ϵ break end
        end
        if xi_rest < ϵ break end
    end

    return (p̄, dot(p̄, z))
end
function worstcase_l1(z::Vector{Float64}, p̄::Vector{Float64}, ξ::Float64)
    @assert maximum(p̄) <= 1 + 1e-9 && minimum(p̄) >= -1e-9 "values must be between 0 and 1"
    @assert ξ >= 0 "xi must be nonnegative"
    @assert length(z) > 0 && length(z) == length(p̄) "z's values needs to be same length as p̄'s values"
    @assert isapprox(sum(p̄), 1, atol=1e-5) "Values of p̄ must sum to one"

    ξ = clamp(ξ, 0, 2)
    size = length(z)
    sorted_ind = sortperm(z)

    out = copy(p̄)
    k = sorted_ind[1]

    ϵ = min(ξ / 2, 1 - p̄[k])
    out[k] += ϵ
    i = size

    while ϵ > 0 && i > 0
        k = sorted_ind[i]
        difference = min(ϵ, out[k])
        out[k] -= difference
        ϵ -= difference
        i -= 1
    end

    return out, dot(out, z)

end