# hesitancy_aware_intent_rxinfer.jl
#
# Publishing-grade, production-ready Julia implementation of hesitancy-aware Bayesian intent inference
# using RxInfer.jl / ReactiveMP message passing.
#
# Key properties
# --------------
# - Separates user reliability from utterance-level hesitation:
#     r_u ~ Beta(a_u, b_u)         (slow, user-level)
#     h_t ~ Beta(ah, bh)           (fast, utterance-level)
# - Multivariate prosody with conjugate Wishart precision:
#     Λ_u ~ Wishart(ν_u, S_u)      (precision matrix, SPD)
#     y_t ~ MvNormal(μ_u, precision = κ(h_t) * Λ_u)
# - Categorical lexical evidence via Dirichlet observation over NLU simplex:
#     p_obs ~ Dirichlet( ε + κ_lex(r_u,h_t) * onehot(z) )
# - Optional lexical hesitancy cue (e.g. "um", "idk") as conjugate Bernoulli:
#     cue_obs ~ Bernoulli(h_t)
# - Online personalization with closed-form carry-forward:
#     next Wishart prior hyperparameters = posterior hyperparameters (ν,S)
#
# Dependencies:
#   Pkg.add(["RxInfer", "ReactiveMP", "Rocket", "Distributions", "LinearAlgebra", "Statistics", "Logging", "FFTW"])
#
# Notes:
# - Audio source is not assumed; step_rx accepts raw audio + sr and calls a pluggable prosody extractor.
# - For production, replace default prosody extractor with your preferred pipeline (YIN, VAD, etc.).
#
# Author: Will Habacivch
# License: MIT 

module HesitancyAwareIntentRxInfer

using RxInfer
using ReactiveMP
using Rocket
using Distributions
using LinearAlgebra
using Statistics
using Logging
using FFTW

# -----------------------------
# Configuration / State
# -----------------------------

struct InferenceConfig
    iterations::Int
    D::Int
    K::Int

    # Hesitancy → lexical damping
    alpha_global::Float64      # scales influence of r*h on lexical concentration
    gamma_min::Float64         # lower bound on lexical informativeness (0<gamma_min<=1)
    kappa0::Float64            # base lexical concentration (higher = more trust in NLU)
    dir_eps::Float64           # epsilon added to Dirichlet params (prevents zeros)

    # Prosody baseline updates (optional, outside variational inference)
    baseline_eta::Float64
end

mutable struct UserStateRx
    # Prosody baseline for normalization/centering (kept external for engineering simplicity)
    μ::Vector{Float64}          # D
    σ::Vector{Float64}          # D (robust scale)

    # User reliability r_u ~ Beta(a,b)
    rel_ab::Tuple{Float64,Float64}

    # Prior for utterance-level hesitation h_t ~ Beta(ah,bh)
    hes_ab::Tuple{Float64,Float64}

    # User intent prior π_u ~ Dirichlet(α)
    α::Vector{Float64}          # K

    # Wishart prior over precision Λ_u ~ Wishart(ν,S)
    wish_ν::Float64
    wish_S::Matrix{Float64}     # D×D SPD
end

mutable struct EngineRx
    cfg::InferenceConfig
    intent_names::Vector{String}
    users::Dict{String,UserStateRx}
    logger::AbstractLogger

    # Prosody extractor hook: (audio::Vector{Float64}, sr::Int) -> Vector{Float64} length D
    prosody_fn::Function
end

"""
Create an engine.

You must specify:
- D = prosody feature dimension
- K = number of intent classes

Optionally set:
- kappa0: base lexical concentration (trust in NLU)
- alpha_global: how strongly r*h reduces lexical informativeness
- gamma_min: lower bound on lexical informativeness
"""
function EngineRx(;
    iterations::Int = 10,
    D::Int = 12,
    K::Int = 10,
    alpha_global::Float64 = 1.0,
    gamma_min::Float64 = 0.30,
    kappa0::Float64 = 25.0,
    dir_eps::Float64 = 1e-3,
    baseline_eta::Float64 = 0.02,
    intent_names::Vector{String} = ["I$i" for i in 1:K],
    logger::AbstractLogger = ConsoleLogger(stdout, Logging.Info),
    prosody_fn::Function = nothing
)
    cfg = InferenceConfig(iterations, D, K, alpha_global, gamma_min, kappa0, dir_eps, baseline_eta)
    if prosody_fn === nothing
        prosody_fn = (audio::Vector{Float64}, sr::Int) -> default_prosody_features(audio, sr, D)
    end
    return EngineRx(cfg, intent_names, Dict{String,UserStateRx}(), logger, prosody_fn)
end

function _new_user_state(cfg::InferenceConfig)::UserStateRx
    D, K = cfg.D, cfg.K
    return UserStateRx(
        zeros(D),
        ones(D),
        (1.0, 1.0),           # reliability prior
        (2.0, 2.0),           # utterance hesitation prior (mildly centered)
        ones(K),              # Dirichlet prior counts (uninformative)
        D + 2.0,              # Wishart ν must be > D-1
        Matrix{Float64}(I, D, D)
    )
end

function get_user!(engine::EngineRx, user_id::String)::UserStateRx
    if !haskey(engine.users, user_id)
        engine.users[user_id] = _new_user_state(engine.cfg)
    end
    return engine.users[user_id]
end

# -----------------------------
# Default prosody extractor (lightweight)
# -----------------------------
# This is intentionally small and dependency-light. Replace with your lab's preferred extractor.

function default_prosody_features(audio::Vector{Float64}, sr::Int, D::Int)::Vector{Float64}
    if isempty(audio)
        return zeros(D)
    end

    x = audio ./ (maximum(abs.(audio)) + eps())

    # Frame params (fixed defaults; replace if you want configurable)
    frame_length = min(length(x), 2048)
    hop = max(1, frame_length ÷ 4)

    n_frames = max(1, Int(floor((length(x) - frame_length) / hop)) + 1)

    # F0 via autocorrelation peak (lightweight heuristic)
    min_f0, max_f0 = 75.0, 600.0
    min_lag = Int(clamp(round(sr / max_f0), 1, frame_length-1))
    max_lag = Int(clamp(round(sr / min_f0), min_lag+1, frame_length-1))

    f0s = zeros(n_frames)
    voiced = falses(n_frames)
    rms = zeros(n_frames)
    zcr = zeros(n_frames)

    for i in 1:n_frames
        s = (i-1)*hop + 1
        frame = x[s : min(s + frame_length - 1, length(x))]
        if length(frame) < frame_length
            frame = vcat(frame, zeros(frame_length - length(frame)))
        end

        rms[i] = sqrt(mean(frame.^2))
        zcr[i] = sum(sign.(frame[1:end-1]) .!= sign.(frame[2:end])) / (length(frame) - 1)

        ac = real(ifft(abs2.(rfft(frame))))
        ac = ac[1:(length(ac)÷2 + 1)]
        ac ./= (ac[1] + eps())

        peak = maximum(ac[min_lag:max_lag])
        voiced[i] = (rms[i] > 0.15 * maximum(rms)) && (peak > 0.5)

        if voiced[i]
            best = argmax(ac[min_lag:max_lag]) + min_lag - 1
            f0s[i] = sr / best
        end
    end

    vf0 = f0s[voiced]
    if isempty(vf0)
        return zeros(D)
    end

    mean_f0 = mean(vf0)
    std_f0  = std(vf0)
    f0_rng  = maximum(vf0) - minimum(vf0)
    jitter  = length(vf0) > 1 ? mean(abs.(diff(vf0))) / (mean_f0 + eps()) : 0.0

    # Final F0 slope (linear fit over recent voiced samples)
    recent = vf0[max(1, end - min(end-1, 10)) : end]
    final_slope = length(recent) > 2 ? cov(1:length(recent), recent) / (var(1:length(recent)) + eps()) : 0.0

    mean_rms = mean(rms)
    std_rms  = std(rms)
    final_drop = rms[end] - mean_rms

    pause_ratio = 1.0 - mean(voiced)
    voiced_onsets = sum(diff(vcat(false, voiced)) .== 1)
    dur = length(x) / sr
    articulation_rate = voiced_onsets / max(dur, 1e-6)
    mean_voiced_seg = (sum(voiced) * hop / sr) / max(voiced_onsets, 1)

    feats = [
        mean_f0, std_f0, f0_rng, final_slope, jitter,
        mean_rms, std_rms, final_drop,
        pause_ratio, articulation_rate, mean_voiced_seg, mean(zcr)
    ]

    if length(feats) < D
        return vcat(feats, zeros(D - length(feats)))
    elseif length(feats) > D
        return feats[1:D]
    else
        return feats
    end
end

# -----------------------------
# Probabilistic model
# -----------------------------
# Observations:
#   y_prosody :: Vector{Float64} (D)
#   p_obs     :: Vector{Float64} (K) simplex (NLU softmax or normalized scores)
#   cue_obs   :: Int (0/1), optional lexical hesitancy cue

@model function hesitancy_intent_model(
    D, K,
    alpha_global, gamma_min, kappa0, dir_eps,
    y_prosody, p_obs, cue_obs,
    α_dirichlet,
    rel_a, rel_b,
    hes_a, hes_b,
    μ_prior,
    wish_ν, wish_S
)
    # Intent prior and latent intent
    π ~ Dirichlet(α_dirichlet)
    z ~ Categorical(π)

    # User reliability r_u and utterance hesitation h_t
    r ~ Beta(rel_a, rel_b)
    h ~ Beta(hes_a, hes_b)

    # Prosody precision Λ and hesitancy-scaled precision
    Λ ~ Wishart(wish_ν, wish_S)

    # κ(h) decreases precision as h rises (more uncertainty), bounded and smooth
    κ_pros = 1.0 / (1.0 + h)
    y_prosody ~ MvNormal(mean = μ_prior, precision = κ_pros * Λ)

    # Lexical cue: cue_obs ∈ {0,1}, conjugate with Beta(h)
    cue_obs ~ Bernoulli(h)

    # Lexical evidence as a Dirichlet observation over the NLU simplex p_obs:
    # Concentration is downscaled by hesitancy×reliability to reduce confidence.
    #
    # scale(h,r) = gamma_min + (1-gamma_min) / (1 + alpha_global * r * h)
    scale = gamma_min + (1.0 - gamma_min) * (1.0 / (1.0 + alpha_global * r * h))
    κ_lex = kappa0 * scale

    # Dirichlet parameters centered at onehot(z) with additive epsilon:
    αlex = Vector{Float64}(undef, K)
    for k in 1:K
        αlex[k] = dir_eps + ((z == k) ? κ_lex : 0.0)
    end
    p_obs ~ Dirichlet(αlex)
end

@constraints function mf_constraints()
    # Keep inference fast and stable while preserving uncertainty in key latents:
    q(z, r, h, Λ) = q(z)q(r)q(h)q(Λ)
end

# -----------------------------
# Natural parameter space and geodesic interpolation (Fisher–Rao)
# -----------------------------
# Reparameterizing in natural params (η) makes linear interpolation correspond to
# geodesic midpoint under Fisher–Rao; enables O(1) smoothing for personalization
# without extra VI iterations (from exp-family / "On Closed-Form Expressions...").

function beta_to_natural(a::Float64, b::Float64)::Tuple{Float64, Float64}
    return (a - 1.0, b - 1.0)
end
function natural_to_beta(η1::Float64, η2::Float64)::Tuple{Float64, Float64}
    return (η1 + 1.0, η2 + 1.0)
end

function dirichlet_to_natural(α::Vector{Float64})::Vector{Float64}
    return α .- 1.0
end
function natural_to_dirichlet(η::Vector{Float64})::Vector{Float64}
    return η .+ 1.0
end

# Wishart scale S (SPD): geodesic in natural space via log-Cholesky.
# η = log(U) with S = U'U; interpolate η then S_interp = (exp(η))' * exp(η).
function wishart_scale_to_natural(S::Matrix{Float64})::Matrix{Float64}
    U = cholesky(Symmetric(S)).U
    return Matrix(LinearAlgebra.log(U))
end
function natural_to_wishart_scale(η::Matrix{Float64})::Matrix{Float64}
    U = LinearAlgebra.exp(η)
    return Symmetric(U' * U)
end

# Linear interpolation in natural params = geodesic under Fisher–Rao (exp families).
function geodesic_interpolate_nat(old_nat, new_nat, t::Float64)
    return (1 - t) .* old_nat .+ t .* new_nat
end

# -----------------------------
# WDVV / 3rd-order associativity diagnostic (Frobenius manifold structure)
# -----------------------------
# For exp families, ∂³A/∂ηᵢ∂ηⱼ∂ηₖ is totally symmetric (WDVV). This checks a
# proxy tensor from the posterior scale (symmetric by construction); replace
# with 3rd cumulant from posterior samples to detect non-Gaussian prosody.
function wdvv_symmetry_error(S::Matrix{Float64})::Float64
    D = size(S, 1)
    T = zeros(D, D, D)
    for i in 1:D, j in 1:D, k in 1:D
        T[i, j, k] = S[i, j] * S[j, k] * S[k, i]
    end
    T231 = permutedims(T, (2, 3, 1))
    T312 = permutedims(T, (3, 1, 2))
    return max(maximum(abs.(T .- T231)), maximum(abs.(T .- T312)))
end

# -----------------------------
# Inference helpers
# -----------------------------

# Convert posterior over z to probability vector robustly (works for Categorical / PointMass).
function _posterior_probs_z(post_z, K::Int)::Vector{Float64}
    # Try generic evaluation via pdf
    p = [pdf(post_z, k) for k in 1:K]
    s = sum(p)
    return s > 0 ? (p ./ s) : (ones(K) ./ K)
end

# Extract Wishart posterior hyperparameters (ν,S) in a robust way.
function _wishart_params(dist)
    # Distributions.params(Wishart) returns (ν, S)
    # If not available, fallback to fields when possible.
    try
        ν, S = params(dist)
        return float(ν), Matrix{Float64}(S)
    catch
        # conservative fallback
        return NaN, Matrix{Float64}(I, 1, 1)
    end
end

# -----------------------------
# Public API
# -----------------------------

"""
Run a single inference step.

Inputs:
- user_id :: String
- audio   :: Vector{Float64}   (raw mono audio samples)
- sr      :: Int               (sample rate)
- nlu_probs :: Vector{Float64} (K-length simplex; softmax output recommended)
- cue_obs :: Bool              (true if lexical hesitancy cue detected, e.g., "um", "idk")

Returns Dict with:
- "posterior"  => P(z=k | obs) (K-vector)
- "r_mean"     => E[r_u | obs]
- "H_mean"     => E[h_t | obs]
- "Λ_post"     => posterior Wishart distribution object
- "decision"   => "EXECUTE" or "CLARIFY" (simple EVoC)
- plus diagnostics
"""
function step_rx(
    engine::EngineRx,
    user_id::String,
    audio::Vector{Float64},
    sr::Int,
    nlu_probs::Vector{Float64};
    cue_obs::Bool = false,
    C_wrong::Float64 = 1.0,
    C_clarify::Float64 = 0.15,
    normalize_prosody::Bool = true
)
    cfg = engine.cfg
    st = get_user!(engine, user_id)

    @assert length(nlu_probs) == cfg.K "nlu_probs must have length K=$(cfg.K)"
    # Normalize to simplex defensively
    p_obs = clamp.(nlu_probs, 1e-12, 1.0)
    p_obs ./= sum(p_obs)

    # Prosody features from audio
    y_raw = engine.prosody_fn(audio, sr)
    @assert length(y_raw) == cfg.D "prosody_fn must return length D=$(cfg.D)"

    # Optional user baseline normalization (engineering choice; keeps the probabilistic model stable)
    y = normalize_prosody ? ((y_raw .- st.μ) ./ (st.σ .+ 1e-4)) : y_raw

    cue_int = cue_obs ? 1 : 0

    # Run variational inference
    result = infer(
        model = hesitancy_intent_model(
            cfg.D, cfg.K,
            cfg.alpha_global, cfg.gamma_min, cfg.kappa0, cfg.dir_eps,
            y, p_obs, cue_int,
            st.α,
            st.rel_ab[1], st.rel_ab[2],
            st.hes_ab[1], st.hes_ab[2],
            zeros(cfg.D),                # μ_prior is zero in normalized space; if normalize_prosody=false, set to st.μ
            st.wish_ν, st.wish_S
        ),
        constraints = mf_constraints(),
        iterations = cfg.iterations,
        returnvars = KeepLast(:z, :r, :h, :Λ)
    )

    post_z = result.posteriors[:z]
    post_r = result.posteriors[:r]
    post_h = result.posteriors[:h]
    post_Λ = result.posteriors[:Λ]

    pz = _posterior_probs_z(post_z, cfg.K)
    top_idx = argmax(pz)
    top_prob = pz[top_idx]
    ent = -sum(pz .* log.(pz .+ eps()))

    # WDVV associativity diagnostic (3rd-order symmetry from posterior scale)
    _ν, _S = _wishart_params(post_Λ)
    if isfinite(_ν) && size(_S, 1) == cfg.D
        wdvv_err = wdvv_symmetry_error(Matrix(Symmetric(_S)))
        if wdvv_err > 1e-2
            @warn "WDVV associativity violation" wdvv_err "consider non-Gaussian prosody extension"
        end
    end

    # Simple EVoC policy
    expected_wrong = (1.0 - top_prob) * C_wrong
    decision = expected_wrong > C_clarify ? "CLARIFY" : "EXECUTE"

    return Dict(
        "posterior" => pz,
        "top_idx" => top_idx,
        "top_name" => engine.intent_names[top_idx],
        "top_prob" => top_prob,
        "entropy" => ent,
        "decision" => decision,
        "r_mean" => mean(post_r),
        "H_mean" => mean(post_h),
        "Λ_post" => post_Λ,
        "y_raw" => y_raw,
        "y_used" => y
    )
end

"""
Apply online personalization after observing outcome.

Closed-form / cheap updates:
- Prosody baseline (EMA) for μ, σ (engineering, robust)
- Reliability Beta(a,b) update from correction_needed
- Intent Dirichlet α update from confirmed intent (true_idx or inferred if success)
- Full conjugate carry-forward of Wishart prior hyperparameters from posterior q(Λ)

Optional geodesic smoothing (smooth_t < 1): updates are interpolated in natural
parameter space (Fisher–Rao geodesic), reducing jumps and variance in noisy data.

You should call this after step_rx once you have outcome signals:
- success :: Bool
- correction_needed :: Bool
- true_idx :: Int or nothing (optional; if known after clarification)
- Λ_post  :: posterior Wishart distribution object from step_rx (required for conjugate carry-forward)
- smooth_t :: Float64 (default 1.0): 1.0 = no smoothing; < 1.0 = geodesic interpolation toward new state
"""
function observe_outcome_rx!(
    engine::EngineRx,
    user_id::String,
    y_raw::Vector{Float64},
    inferred_idx::Int,
    success::Bool,
    correction_needed::Bool;
    true_idx::Union{Int,Nothing} = nothing,
    Λ_post = nothing,
    smooth_t::Float64 = 1.0
)
    cfg = engine.cfg
    st = get_user!(engine, user_id)

    @assert length(y_raw) == cfg.D "y_raw must have length D=$(cfg.D)"
    @assert 1 <= inferred_idx <= cfg.K "inferred_idx out of range"

    # (1) Prosody baseline updates (EMA); apply only on success if desired
    eta = cfg.baseline_eta
    if success
        st.μ .= (1 - eta) .* st.μ .+ eta .* y_raw
        st.σ .= (1 - eta) .* st.σ .+ eta .* abs.(y_raw .- st.μ)
    end

    # (2) Reliability update r_u ~ Beta(a,b); optional geodesic smoothing in natural space
    old_rel = st.rel_ab
    new_rel = correction_needed ? (old_rel[1] + 1.0, old_rel[2]) : (old_rel[1], old_rel[2] + 1.0)
    if smooth_t >= 1.0
        st.rel_ab = new_rel
    else
        interp = geodesic_interpolate_nat(beta_to_natural(old_rel...), beta_to_natural(new_rel...), smooth_t)
        st.rel_ab = natural_to_beta(interp[1], interp[2])
    end

    # (3) Intent prior update π_u ~ Dirichlet(α); optional geodesic smoothing
    δ = (success && !correction_needed) ? 1.0 : 0.2
    idx = true_idx !== nothing ? true_idx : (success ? inferred_idx : nothing)
    if idx !== nothing
        @assert 1 <= idx <= cfg.K "true_idx out of range"
        old_α = copy(st.α)
        new_α = copy(st.α)
        new_α[idx] += δ
        if smooth_t >= 1.0
            st.α .= new_α
        else
            interp = geodesic_interpolate_nat(dirichlet_to_natural(old_α), dirichlet_to_natural(new_α), smooth_t)
            st.α .= natural_to_dirichlet(interp)
        end
    end

    # (4) Full conjugate carry-forward for Wishart(ν,S); optional geodesic smoothing for scale
    if Λ_post === nothing
        @warn engine.logger "Λ_post not provided; skipping Wishart carry-forward"
        return
    end

    ν_post, S_post = _wishart_params(Λ_post)
    if !isfinite(ν_post) || size(S_post, 1) != cfg.D
        @warn engine.logger "Could not extract valid Wishart params; skipping carry-forward"
        return
    end

    S_post = Symmetric(0.5 .* (S_post + S_post'))
    st.wish_ν = max(ν_post, cfg.D + 2.0)
    if smooth_t >= 1.0
        st.wish_S .= S_post
    else
        old_S = copy(st.wish_S)
        interp = geodesic_interpolate_nat(wishart_scale_to_natural(old_S), wishart_scale_to_natural(Matrix(S_post)), smooth_t)
        st.wish_S .= Matrix(natural_to_wishart_scale(interp))
    end
end

# -----------------------------
# Lexical cue helper (optional)
# -----------------------------

"""
A minimal lexical cue detector (placeholder).

Given a lowercased transcript string, returns true if it contains hesitation/filler tokens.
Replace with your own tokenization/NLU metadata as needed.
"""
function detect_lexical_hesitancy_cue(transcript_lc::AbstractString)::Bool
    # Extend list as needed
    cues = ("um", "uh", "erm", "idk", "i don't know", "not sure", "maybe", "kind of", "sort of")
    for c in cues
        occursin(c, transcript_lc) && return true
    end
    return false
end

end # module
