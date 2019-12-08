import graphsage.model as model
import numpy as np


if __name__ == "__main__":

    acc = np.zeros(10)
    times = np.zeros(10)
    for seed in range(10):
        acc[seed], times[seed] = model.run_reddit(seed)

    print(np.mean(acc), np.mean(times))
    print(np.std(acc), np.std(times))
