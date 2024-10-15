function worst_case_l1(z::Vector{Float64}, pbar::Vector{Float64}, xi::Float64)
    @assert maximum(pbar) <= 1 + 1e-9 && minimum(pbar) >= -1e-9 "values must be between 0 and 1"
    @assert xi >= 0 "xi must be nonnegative"
    @assert length(z) > 0 && length(z) == length(pbar) "z's values needs to be same length as pbar's values"

    xi = clamp(xi, 0, 2)
    size = length(z)
    sorted_ind = sortperm(z)

    out = copy(pbar) #duplicate it
    k = sorted_ind[1] #index begins at 1

    epsilon = min(xi / 2, 1 - pbar[k])
    out[k] += epsilon
    i = size -1

    while epsilon > 0 && i > 0
        k = sorted_ind[i]
        i -= 1
        difference = min(epsilon, out[k])
        out[k] -= difference
        epsilon -= difference
    end

    return out, dot(out, z)

end

    