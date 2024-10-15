from Functions import *

from matplotlib import pyplot as plt

import numpy as np

iterations = 50000


def main():
    # Dictionary of all variables to be loaded
    results = {"renders": 0,
               "rewards": 0, 
               "dones": 0, 
               "truncateds": 0,
               "times": 0}
    
    # Loading arrays for all the metrics
    path_to_results = results_path()
    for var in results.keys():
        for file in os.listdir(path_to_results):
            if var in file:
                print(f"{var} loaded")
                results[var] = np.load(os.path.join(path_to_results, file), allow_pickle=True)

    print(results)

        



if __name__=="__main__":
    main()