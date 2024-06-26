import numpy as np


def PoissonProcess_simulation(rate, num_events):
    # Generation of exponential inter-arrival times
    inter_arrival_times = np.random.exponential(scale=1/rate, size=num_events)
    # Actual arrival times
    effective_arrival_times = np.cumsum(inter_arrival_times)

    return effective_arrival_times
