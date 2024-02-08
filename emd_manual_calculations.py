import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import pulp

 
# manually calculate emd using linear programming
def emd_manual(a, b):

    # set up pulp linear programming problem
    emd_problem = pulp.LpProblem('EMD', sense = 1)

    # Count occurrences of each unique value
    a_unique, a_counts = np.unique(a, return_counts=True)
    b_unique, b_counts = np.unique(b, return_counts=True)

    # convert to probability distribution
    prob_dist_a = a_counts / np.sum(a)
    prob_dist_b = b_counts / np.sum(b)

    # normalize probability distribution
    norm_a = prob_dist_a / np.sum(prob_dist_a)
    norm_b = prob_dist_b / np.sum(prob_dist_b)

    n = len(norm_a)
    m = len(norm_b)

    # Define decision variables
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(m)],
                              lowBound=0, cat='Continuous')

    # Add objective function (minimize total cost)
    emd_problem += pulp.lpSum(x[i, j] * np.abs(a[i] - b[j])
                              for i in range(n) for j in range(m))       

    ## Constraints
    for i in range(n):
        emd_problem += pulp.lpSum(x[i, j] for j in range(m)) == norm_a[i]

    for j in range(m):

        emd_problem += pulp.lpSum(x[i, j] for i in range(n)) == norm_b[j]

    emd_problem.solve()

    if emd_problem.status == 1:
        print('solution found!')
    else:
        print('no feasible solution found')

    emd_calc = pulp.value(emd_problem.objective)

    # put together transport plan
    transport_plan = {(i, j): pulp.value(x[i, j]) for i in range(n) for j in range(m)}
    plan_values = [row.varValue for row in emd_problem.variables()]

    # adjust transport plan for normalization
    for key in transport_plan:

        transport_plan[key] *= len(a)
        transport_plan[key] = round(transport_plan[key])

    return emd_calc, transport_plan, plan_values

 

if __name__ == "__main__":

    a =  [1, 1, 2, 2, 2, 3, 3, 3]
    b = [6, 6, 7, 8, 8, 8, 9, 9]

    dist, plan, plan_values = emd_manual(a, b)
    print(dist)
    print(plan)