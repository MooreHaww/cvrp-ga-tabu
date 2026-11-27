# Complexity validation: log-log slopes

| component | x | theory | theory_slope | fit_all_slope | fit_all_se | fit_all_r2 | fit_n100_slope | fit_n100_se | fit_n100_r2 | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cw | n | O(n^2 log n) | 2.179 | 2.378 | 0.059 | 0.9981 | 2.450 | 0.074 | 0.9982 | differs | the log factor puts the expected slope a bit above 2; the n^2 savings list also stresses the cache |
| split_unlimited | n | O(n L), L constant | 1.000 | 1.107 | 0.013 | 0.9996 | 1.121 | 0.017 | 0.9995 | consistent | L is constant (Q fixed); per-call overhead (tour check, prefix sums) flattens small n |
| split_k | n | O(K n L), K ~ n | 2.000 | 1.962 | 0.043 | 0.9985 | 2.019 | 0.049 | 0.9988 | consistent | K grows linearly with n because Q is fixed, so K*n*L is quadratic |
| ga_generation | n | O(P n L) | 1.000 | 1.961 | 0.042 | 0.9986 | 2.013 | 0.050 | 0.9987 | differs | offspring needing more than K routes also run the K-limited Split, O(K n L) ~ n^2, even when it fails; arcs_per_decode counts successful K-limited splits only, so it does not show this fallback |
| gals_generation | n | O(P (n L + passes n kappa)) | 1.000 | 1.479 | 0.026 | 0.9991 | 1.518 | 0.022 | 0.9996 | differs | dominated by the descent, whose number of passes grows with n; decode may also fall back to the K-limited Split |
| tabu_iteration | n | O(n kappa) | 1.000 | 1.140 | 0.027 | 0.9983 | 1.183 | 0.020 | 0.9995 | differs | candidate scan O(n kappa) plus O(K + L) to apply the move and update the tabu list |
| tabu_delta | n | O(1) per move | 0.000 | 0.197 | 0.041 | 0.8870 | 0.252 | 0.045 | 0.9406 | differs | the delta does not depend on n: the line should be flat |
| tabu_full_eval | n | O(n) per move | 1.000 | 0.997 | 0.022 | 0.9986 | 1.028 | 0.021 | 0.9992 | consistent | copying the routes and re-summing every route is O(n) |
| descent | n | O(passes n kappa) | 1.000 | 1.337 | 0.079 | 0.9898 | 1.426 | 0.103 | 0.9898 | differs | the number of passes is empirical and grows with n, so the total grows faster than one pass |
| descent_per_pass | n | O(n kappa) | 1.000 | 1.078 | 0.025 | 0.9984 | 1.068 | 0.042 | 0.9970 | consistent | one pass scans n customers x kappa neighbours |
| split_unlimited_q | L | O(n L) at fixed n | 1.000 | 0.578 | 0.066 | 0.9621 | – | – | – | differs | linear in L (Q grows, n fixed) |
| split_k_q | L | O(K n L), K ~ 1/Q | 0.078 | -0.035 | 0.026 | 0.3879 | – | – | – | consistent | K*L is about constant: K shrinks as Q grows |

Fit: log10(time) = a + b*log10(x) by least squares on the median of the repetitions per size; fit_n100 uses n >= 100 only (small n are dominated by constant per-call overhead). theory_slope = slope of the theoretical cost over the same sizes (n^2 log n gives a bit more than 2). verdict = consistent if |slope - theory_slope| <= max(2*SE, 0.15) (n-components: the n >= 100 fit; Q-sweep rows: the fit over all Q), else differs. The 0.15 tolerance covers timer noise, cache effects and the few sizes; SE, slope and R^2 are reported unchanged.
