# GECCO 2026: ML4VRP Competition - CVRP Track
- **Goal:** Solve the Capacitated Vehicle Routing Problem (CVRP).
- **Objective Function:** `1000 * NV + TD` (NV = Number of Vehicles, TD = Total Distance).
- **Dataset:** X-dataset (Uchoa et al., 2014), instances with 100-400 customers.
- **Priority:** Fleet minimization (NV) is the primary target due to the 1000x multiplier.
- **Constraints:** All routes must start/end at depot (node 1), and total demand must not exceed capacity Q.