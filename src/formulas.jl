
# --------------------------------------------------------------------------------------------------------
# ANALYTICAL FORMULAS

function alpha_SHO(a, ω0, ν0, η)
    exp_term = exp(-a / (2ν0))

    if η < 2
        ρ = sqrt(1 - (η/2)^2)
        θ = ω0 * ρ / (ν0)
        trig_term = cos(θ) + (a / (2ω0 * ρ)) * sin(θ)
    elseif η==2
        ρ = sqrt(1 - ((η-10e-4)/2)^2)
        θ = ω0 * ρ / (ν0)
        trig_term = cos(θ) + (a / (2ω0 * ρ)) * sin(θ)
    else
        ρ_tilde = sqrt(-(1 - (η/2)^2))
        θ = ω0 * ρ_tilde / (ν0)
        trig_term = cosh(θ) + (a / (2ω0 * ρ_tilde)) * sinh(θ)
    end
    return 1 - exp_term * trig_term
end

function rho_SHO(η, ω0, μ_λ,n)
    γ = η * ω0 / μ_λ  .* log(2)
    return 0.5 * (1 - exp.(-γ))^2 / (γ - (1 - exp.(-γ))) .* exp(-(n-1) *γ)
end



function ksho(t::Real, η::Real, ω0::Real)
    @assert ω0 > 0 "ω0 must be positive"
    τ = abs(t)
    ρ = sqrt(abs(1 - (η)^2))

    # common exponential damping factor
    damp = exp(-(η * ω0 ) * τ)

    C = if η < 1 - 1e-12                     # underdamped
        cos(ρ * ω0 * τ) + (η / ρ) * sin(ρ * ω0 * τ)
    elseif η > 1 + 1e-12                     # overdamped
        cosh(ρ * ω0 * τ) + (η / ρ) * sinh(ρ * ω0 * τ)
    else                                     # critically damped (η ≈ 1)
        2 * (1 + ω0 * τ)
    end

    return (1 / ω0^2) * damp * C
end

# Broadcast over arrays of t
function ksho(t::AbstractArray, η::Real, ω0::Real)
    return ksho.(t, η, ω0)
end

function psd_SHO(ω, S0, Q, ω0)
    numerator = sqrt(2 / π) * S0 * ω0^4
    denominator = (ω^2 - ω0^2)^2 + (ω^2 * ω0^2) / Q^2
    return numerator / denominator
end
