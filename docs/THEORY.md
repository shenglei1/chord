# Theory obligations for the proposed method

The proposed implementation is named **CHORD-CAHR** (Coalition Hypergraph for
Open Robotic Dispatch with Capacity-Aware Hyperedge Reservation). It is trained
with a joint conditional MAPPO objective. The statements below concern its representation and
the open warehouse decision process; they do not convert the empirical EAAI
hypothesis into a theorem of algorithmic superiority.

The implementation provides the structural objects needed by the theory. The
paper must state assumptions and prove the results below.

## 1. Random-finite-set open warehouse game

At decision step `t`, let the state be

```
X_t = (A_t, T_t, R_t, H_t),
```

where `A_t` is the typed agent set, `T_t` the order set, `R_t` shared
resources, and `H_t` the feasible service hypergraph. Birth/death kernels must
be Markov conditional on `X_t` and the exogenous demand mode.

## 2. Permutation equivariance

For type-preserving agent permutation `P_A` and task permutation `P_T`, prove

```
pi(P_A X, P_T X, P H) = P_A pi(X, H) P_T^T.
```

The proof follows from shared type encoders, symmetric incidence aggregation,
shared coalition scoring, permutation-invariant capacity/load descriptors and
invariant masked pooling. Under type-preserving relabeling, candidate coalition
categories are relabelled and the autoregressive mask applies the same
resource/task exclusion relation. Therefore the induced distribution over
conflict-free coalition sets is equivariant. This statement assumes unique
physical top-k distances or exchangeability of exact-distance ties; a slot-ID
tie-break cannot be used as semantic information.

Task-coalition membership is the intersection of type compatibility, current
activity and the observable action-feasibility mask. Consequently a busy agent
cannot appear as an executable candidate merely because it is spatially close.

## 3. Conflict-free reservation invariant

Let `c_l=(a_l,p_l,t_l)` be the coalition selected at reservation round `l`.
After each selection, CAHR masks every remaining candidate sharing `a_l`,
`p_l`, or `t_l`. By induction, for the executed set `C`:

```
sum_{c in C} 1[a(c)=a] <= 1,
sum_{c in C} 1[p(c)=p] <= 1,
sum_{c in C} 1[t(c)=t] <= 1.
```

Thus same-role duplicate assignment to an order is impossible by construction,
not merely discouraged in expectation. Feasibility still depends on the input
action mask, so the theorem is conditional on a correct environment mask.

## 4. Joint conditional MAPPO surrogate

For a stored reservation sequence terminated by STOP, the policy likelihood is

```
log pi(C|X) = sum_l m_l log pi(c_l | X, c_<l, M_l),
```

where `M_l` is the stored conditional feasibility mask and `m_l` excludes
padding rounds. PPO uses the exact joint ratio

```
r = exp(log pi_new(C|X) - log pi_old(C|X)).
```

This avoids multiplying unrelated marginal proposals after an external
conflict resolver. It makes the optimized policy identical to the executed
autoregressive decoder. PPO clipping remains a local surrogate and does not
imply global neural-PPO convergence.

## 5. Bellman well-posedness

On bounded functions over random finite sets, with bounded reward and
`gamma < 1`, prove the standard sup-norm contraction

```
||T Q_1 - T Q_2||_inf <= gamma ||Q_1 - Q_2||_inf.
```

This establishes existence and uniqueness of the optimal value for the model;
it does **not** establish global convergence of a neural PPO optimizer.

## 6. Population-transfer bound

Assume rewards are `L_r`-Lipschitz in the empirical entity measure, feasible
hyperedge sets vary Lipschitz-continuously in Hausdorff distance, and the
controlled population transition is `kappa`-contractive in Wasserstein-1 with
`gamma * kappa < 1`. Prove a value regularity result of the form

```
|V*(mu) - V*(nu)| <= L_r / (1 - gamma*kappa) * W_1(mu, nu)
                    + feasible-set perturbation term.
```

This result, not variable tensor length by itself, supports the unseen-team-size
claim. If the transition is not contractive, provide a finite-horizon bound
instead of overstating the result.

## 7. Complexity

With `N` agents, `M` tasks, per-role shortlist `k`, `L` reservation rounds and
`Z` zones, candidate construction costs `O(MN + Mk^2)`, hypergraph message
passing costs `O(kMd + Z(N+M)d)`, and CAHR scoring/masking costs
`O(Mk^2d_r + LMk^2)` for reservation width `d_r`. Scores are computed once;
only Boolean availability masks change across rounds. The implementation uses
batched tensor construction and projects entity embeddings before gathering
padded candidates to bound accelerator memory.
