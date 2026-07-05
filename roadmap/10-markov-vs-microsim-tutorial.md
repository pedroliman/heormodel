# 10. Markov vs microsimulation models (cross-validation tutorial)

A tutorial titled "Markov vs microsimulation models" that does two things at once: cross-validate the two engines against each other, and show what a microsimulation can represent that a cohort Markov model cannot. The organizing theme is risk heterogeneity. It ships as `examples/markov_vs_microsim.py` and a website tutorial, and it doubles as a cross-engine validation test.

This item assumes items 3 and 5 (the microsimulation and Markov engines) and reads best after the Markov tutorial in the docs order set by item 11.

## The argument the tutorial makes

Build one Sick-Sicker-style model twice: as a `MarkovModel` cohort trace and as a `MicrosimModel` individual simulation, from the same transition probabilities, rewards, and horizon.

1. Identical assumptions converge. With a memoryless, homogeneous population, the microsimulation mean cost and QALY converge to the cohort trace as the population grows. Show the convergence: plot the microsimulation mean against the cohort value as `n_individuals` rises, and report the gap at a large population. This is the cross-validation: two independent implementations agree, so both are trusted.

2. Heterogeneity breaks the cohort assumption. Give each individual a frailty multiplier on their progression and mortality hazards, drawn from a distribution with the same mean as the cohort rate. The marginal transition probability is unchanged, yet the microsimulation mean survival and QALYs now differ from the cohort trace, because accrual is non-linear in the hazard and correlates with how long an individual lives. Quantify the divergence and explain it: a cohort model tracks the average person, and the average of a non-linear function is not the function of the average. This is the risk-heterogeneity point, made concrete with a number.

3. What else the microsimulation buys. Briefly, with the same machinery: history dependence (a mortality risk that rises with time spent sick, using `duration_groups`), and individual-level cost caps or one-time events that a cohort cannot carry. These are the assumptions a modeler can relax once the population is individual-level. Keep this section short; heterogeneity is the headline.

The tutorial states the trade plainly. The cohort model is faster and exact for its assumptions; the microsimulation costs iterations but represents heterogeneity and history the cohort averages away. Neither is more correct in general; they answer under different assumptions.

## Deliverables

- `examples/markov_vs_microsim.py`: builds both models, runs the convergence sweep, adds frailty, prints the cohort-versus-microsim comparison table, and saves the convergence plot and the heterogeneity comparison.
- A website tutorial narrating it, placed between the Markov and microsimulation tutorials (item 11), following `guidance/writing_style.md`.

## Acceptance

- A test asserts the homogeneous microsimulation mean converges to the cohort trace within tolerance at a large population, at a fixed seed (the cross-validation).
- A test asserts the heterogeneous microsimulation mean differs from the cohort by more than Monte Carlo noise, in the direction the non-linearity predicts, so the divergence is a property of the model and not a bug.
- The example and the rendered tutorial run under `uv run python` and `uv run quartodoc build`, and the prose matches the printed numbers and plots.
