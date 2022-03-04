from matplotlib import pyplot as plt
import numpy as np

from efbp import run_simulation


def detect_cycle(hist, memory):
    """
    return the cycle length if there is a cycle longer 2*memory
    
    Repeated patterns of length less than 2*memory may not repeat
    indefinitely as attendance at each timestep relies on the previous
    2 * memory timesteps.
    """
    l = 2 * memory # minimum cycle length

    tail = hist[-l :]

    # we must leave space for an entire pattern repeat
    # of length *l* before the beginning of the tail
    for i in range(len(hist) - 2*l -1, 0, -1):
        if (hist[i:i+l] == tail).all():
            return len(hist) - l - i
    return None


def summarize_run(**kwargs):
    
    hist, best_strat, pred_history = run_simulation(**kwargs)

    plt.figure()
    plt.title("Weekly Attendance")
    plt.xlabel("Time")
    plt.ylabel("Number attending")
    plt.plot(hist)
    plt.plot([-1, len(hist)+1], [60, 60], "--")

    
    plt.figure()
    # disregard the first 100 weeks to give the system
    # time to settle
    plt.hist((pred_history[:, 100:] < 60).sum(axis=1) / (len(hist) - 50), bins=20)
    plt.title("Attendance Rates")
    plt.xlabel("Rate")
    plt.ylabel("Agent Count")
    
    plt.figure()
    plt.title("Strategies Used per Agent")
    plt.xlabel("Number of strategies")
    plt.ylabel("Agent Count")
    
    # once again, disregard the first 100 weeks
    steady_state_strats = np.apply_along_axis(
        lambda a: len(np.unique(a)),
        1,
        best_strat[:, 100:],
    )

    plt.hist(steady_state_strats, bins=steady_state_strats.max())
    
    memory = kwargs.get("memory", 8)
    cl = detect_cycle(hist, memory)
    print("Cycle length: {}".format(cl))
    
    if cl:
        print(hist[-cl:])