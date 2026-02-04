#!/usr/bin/env julia
# Minimal test for HesitancyAwareIntentRxInfer: step_rx, observe_outcome_rx! with geodesic smoothing.
#
# From H_Aware directory (with Julia deps installed):
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'
#   julia --project=. test_hesitancy_aware.jl

using Test
using Logging
global_logger(NullLogger())  # quiet RxInfer

include(joinpath(@__DIR__, "Hesitancy_Aware.jl"))
using .HesitancyAwareIntentRxInfer

function main()
    D, K = 4, 3
    engine = EngineRx(; D = D, K = K, iterations = 4)

    # Fake 0.5s mono @ 16kHz
    sr = 16000
    audio = randn(sr ÷ 2) .* 0.1
    nlu_probs = [0.7, 0.2, 0.1]  # simplex

    # Step 1: infer
    result = step_rx(engine, "user1", audio, sr, nlu_probs; cue_obs = false)
    @test haskey(result, "posterior")
    @test haskey(result, "Λ_post")
    @test length(result["posterior"]) == K
    inferred_idx = result["top_idx"]
    y_raw = result["y_raw"]

    # Step 2: personalize with geodesic smoothing (smooth_t = 0.5)
    observe_outcome_rx!(
        engine, "user1", y_raw, inferred_idx, true, false;
        Λ_post = result["Λ_post"],
        smooth_t = 0.5,
    )

    # Step 3: second step to confirm state is consistent
    result2 = step_rx(engine, "user1", audio, sr, nlu_probs; cue_obs = false)
    @test haskey(result2, "posterior")
    @test length(result2["posterior"]) == K

    # Sanity: natural param round-trip
    a, b = 2.0, 3.0
    η1, η2 = HesitancyAwareIntentRxInfer.beta_to_natural(a, b)
    a2, b2 = HesitancyAwareIntentRxInfer.natural_to_beta(η1, η2)
    @test a2 ≈ a && b2 ≈ b

    println("OK: step_rx, observe_outcome_rx!(smooth_t=0.5), natural param round-trip")
    return true
end

main()
