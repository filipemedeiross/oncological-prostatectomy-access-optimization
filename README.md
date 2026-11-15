# ONCOLOGICAL PROSTATECTOMY ACCESS OPTIMIZATION

Analysis of hospital admission flows in Brazil's SUS and reallocation scenarios to reduce travel between municipalities of residence and care. The main experiments cover **January–June 2026**.

## Data and Experiments

SIH/SUS data processing selects procedure `0416010121`, regular hospital admission authorizations (AIHs, `IDENT = 1`), and admission dates (`DT_INTER`), with a processing cutoff of September 2026. The unit of analysis is the admission, rather than a unique patient. Demand represents observed care; each provider municipality's observed production serves as a capacity proxy.

| Experiment | Purpose |
| --- | --- |
| [01 — Data Processing](notebooks/01_sih_data_processing.py) | Extract, filter, and consolidate admissions. |
| [02 — Municipal Analysis](notebooks/02_municipal_data_analysis.py) | Aggregate demand and production and analyze their geographic distribution. |
| [03 — OD Matrix](notebooks/03_origin_destination_data.py) | Assign distances to municipality pairs and export the optimization instance. |
| [04 — Observed Flows](notebooks/04_observed_flow_analysis.py) | Describe recorded travel through indicators and maps. |
| [05 — Nearest Provider](notebooks/05_nearest_provider_reallocation.py) | Scenario 1: reallocate demand without capacity constraints. |
| [06 — Capacity Constraint](notebooks/06_capacity_constrained_reallocation.py) | Scenario 2: minimize travel under hypothetical capacity expansion. |
| [07 — Interstate Limit](notebooks/07_capacity_constrained_reallocation_with_interstate_limit.py) | Scenario 3: limit admissions outside the state of residence. |
| [08 — Historical Analysis](notebooks/08_historical_flow_analysis.py) | Compare annual scenarios and estimate potential transport emissions. |

> **Note**.: Distances use the Saldanha matrix and its May 2024 reference network, including in the historical analysis. Travel within a municipality is assigned zero distance.

## Mathematical Models

Let $`I`$ be the municipalities of residence, $`J`$ the municipalities with observed production, $`D_i`$ the demand at $`i`$, $`U_j^{obs}`$ the production at $`j`$, and $`d_{ij}`$ the distance in kilometers. The variable $`x_{ij}`$ represents the number of admissions reallocated from $`i`$ to $`j`$. All scenarios preserve demand and keep the same municipalities eligible to provide care.

### Scenario 1 — Nearest provider

Each municipality assigns all its demand to the nearest provider, without capacity constraints:

```math
\begin{aligned}
j^*(i) &= \arg\min_{j\in J} d_{ij},\\
x_{ij} &= \begin{cases}
D_i, & j=j^*(i),\\
0, & \text{otherwise}.
\end{cases}
\end{aligned}
```

This scenario yields the minimum admission-weighted distance for the fixed provider set.

### Scenario 2 — Capacity-constrained transportation

Expansion $`s\geq0`$ is applied individually to each provider municipality's observed production. The model minimizes total admission-weighted distance:

```math
\begin{aligned}
\min_x\quad & C(x)=\sum_{i\in I}\sum_{j\in J}\tilde d_{ij}x_{ij}\\
\text{subject to}\quad &\sum_{j\in J}x_{ij}=D_i &&\forall i\in I,\\
&\sum_{i\in I}x_{ij}\leq\lfloor(1+s)U_j^{obs}\rfloor &&\forall j\in J,\\
&x_{ij}\geq0 &&\forall i\in I,\ j\in J.
\end{aligned}
```

This is a transportation linear program with integer margins; the basic solution obtained by simplex is integral, and integrality is checked before export.

### Scenario 3 — Interstate limit

This scenario retains Scenario 2's objective and demand and capacity constraints, adds a national cap $`q`$ on the share of admissions outside the state of residence, and requires integer flows:

```math
\begin{aligned}
\sum_{\substack{i\in I,\ j\in J\\state(i)\neq state(j)}}x_{ij}
&\leq\left\lfloor qN\right\rfloor,\\
N&=\sum_{i\in I}D_i,\qquad x_{ij}\in\mathbb{Z}_{\geq0}.
\end{aligned}
```

As a counterfactual, it computes the smallest interstate share among distance-optimal solutions. Let $`F_s`$ contain the integer allocations satisfying demand and capacity constraints, $`K(x)`$ count interstate admissions, and $`Z_s^*=\min_{x\in F_s}C(x)`$. It then solves:

```math
\begin{aligned}
K_{\min}&=\min_{x\in F_s}\{K(x):C(x)\leq\alpha Z_s^*\},\\
q_{\min}&=K_{\min}/N,\qquad \alpha=1.
\end{aligned}
```

> **Note**.: The transportation models use **Pyomo and HiGHS**, with costs $`\tilde d_{ij}=rint(d_{ij})`$ (rounding to integer kilometers, with ties to the nearest even integer).
