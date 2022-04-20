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
        
        
def err(hist, a):
    [c_4, c_3, c_2, c_1] = hist
    return (c_3 - c_1 + (c_2 - c_3) * a)**2 + (c_4 - c_2 + (c_3 - c_4) * a)**2


def pred(hist, a):
    [c_4, c_3, c_2, c_1] = hist
    return a * c_1 + (1 - a) * c_2


def plot_errs(hist):
    x = np.arange(-1, 1.01, 0.01)
    y = err(hist, x)
    going = pred(hist, x)

    mask = np.where(going < 60)
    plt.plot(x[mask],y[mask], color="blue")

    mask = np.where(going >= 60)
    plt.plot(x[mask],y[mask], color="orange")

    plt.xlabel("a")
    plt.ylabel("Error")
    plt.legend(["going", "not going"])
    _ = plt.title(hist)
    
    
def binom_p_value(k, n, p):
    if k < n * p:
        return binom.cdf(k, n, p)
    else:
        return binom.sf(k, n, p)
    
    

def assert_equal(a, b, msg):
    assert (np.abs(a - b) < 1e-6).all(), msg
    

def assert_less(a, b, msg):
    assert (a - b < 1e-6).all(), msg

    
def assert_geq(a,b,msg):
    assert (a - b > -1e-6).all(), msg
    