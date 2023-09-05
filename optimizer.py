import cvxpy as cp
import numpy as np


def optimize(data, capacity, kwh_per_km):
    """Return optimal charging time series. 

    Args:
        data (DataFrame): contains information about emissions level and distance traveled for each charger that is reached
        capacity (float): battery capacity of EV (kWh)
        kwh_per_km (float): amount of enegy it takes the EV to go one km

    Returns:
        numpy array: item i in the array represents a kWh of power drawn from the ith charging station encountered
    """

    x = cp.Variable(len(data))
    a = np.array(data['emissions'])
    b = np.repeat(capacity, len(a))
    w = np.array(data['cumdist'])
    M = np.tril(np.ones((len(w), len(w))), k=-1)

    objective = cp.Minimize(a @ x)
    entry_lb = M @ x + b - w * kwh_per_km >= 0
    entry_ub = M @ x + b - w * kwh_per_km <= capacity
    charge_lb = x >= 0
    charge_ub = x <= 30
    
    constraints = [entry_lb, entry_ub, charge_lb, charge_ub]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="SCIPY")
    return x.value

