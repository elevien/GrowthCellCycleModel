
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

# WIP
function W_theory(η, ω0,ν0,l)
    # compute What matrix 
    a = 2*η * ω0
    M = [0. 0. 1.0 ; 0. 0. 1.0;  0.0 -ω0^2  -a ]
    What = exp(M/ν0)
    βϕ = zeros(l)
    βz = zeros(l+1)
    βz[1] = What[1,2]
    for i in 1:l
        βϕ[i] = What[1,3]*What[3,3]^i
        βz[i+1] = What[1,3]*What[3,3]^(i-1)*What[3,2]
    end
    return βϕ,βz
end

function k_SHO(τ, S0, Q, ω0)
    η = sqrt(abs(1 - 1 / (4Q^2)))  # η is defined differently for different Q ranges

    if 0 < Q && Q < 1/2
        damping_factor = cosh(η * ω0 * τ) + (1 / (2 * η * Q)) * sinh(η * ω0 * τ)
    elseif Q == 1/2
        damping_factor = 2 * (1 + ω0 * τ)
    elseif Q > 1/2
        damping_factor = cos(η * ω0 * τ) + (1 / (2 * η * Q)) * sin(η * ω0 * τ)
    else
        error("Q must be positive.")
    end

    return S0 * ω0 * Q * exp(-ω0 * τ / (2Q)) * damping_factor
end

function psd_SHO(ω, S0, Q, ω0)
    numerator = sqrt(2 / π) * S0 * ω0^4
    denominator = (ω^2 - ω0^2)^2 + (ω^2 * ω0^2) / Q^2
    return numerator / denominator
end
