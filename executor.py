from trainHighwayEnvDQN import main as Highway
from trainHiMeEnvDQN import main as HiMe
from trainMergeEnvDQN import main as Merge
from accessModelHighway import main as accessHighway
from accessModelMerge import main as accessMerge
from trainHighwayEnvCriticalsDQN import main as trainCritHi
from trainMergeEnvCriticalsDQN import main as trainCritMe
import time, numpy

for i in range(10):
    Highway(100000, i, 100) # Train highway
    print("highway trained")
    Merge(100000, 10000, i, 100) # Train Merge 10000 on model of Highway 100000
    print("merge trained")
    accessMerge(10000, i, 100, i) # Access Merge 10000 on merge-env
    print("merge accessed")
    accessMerge(100000, i, 100, i) # Access Highway 100000 on merge-env to get criticals
    print("highway accessed")
    trainCritMe(i, 100000, f"model_dqn_100000({i})", 100, i, 1000) # train Merge on criticals from highway 100000 in merge-env
    print("merge crit trained")
    accessMerge(1000, i, 100, i) # Access Merge trained on criticals
    print("merge crit accessed")


# start = time.time()
# trainHi(1000, 5)
# trainCritHi()
# crit = time.time() - start

# start = time.time()
# trainHi(100000, 0)
# simple = time.time() - start
# Highway(iterations=100000, copy_num=1)
# trainCritHi(copy_num=6)
# accessHighway(10000, 6, 300, 0)
# # accessHighway(100000, 1, 300, 0)
# accessHighway(10000, 6, 3000, 1)
# # accessHighway(100000, 1, 1500, 1)
# accessHighway(10000, 6, 30000, 2)
# #accessHighway(100000, 1, 2600, 2)
# accessHighway(10000, 6, 300000, 3)
# #accessHighway(100000, 1, 3700, 3)
# accessHighway(10000, 6, 3000000, 4)
# #accessHighway(100000, 1, 4800, 4)

# Highway(iterations=100000, copy_num=2)
# trainCritHi(copy_num=5)
# accessHighway(10000, 5, 300, 0)
# accessHighway(100000, 2, 300, 0)
# accessHighway(10000, 5, 1500, 1)
# accessHighway(100000, 2, 1500, 1)
# accessHighway(10000, 5, 2600, 2)
# accessHighway(100000, 2, 2600, 2)
# accessHighway(10000, 5, 3700, 3)
# accessHighway(100000, 2, 3700, 3)
# accessHighway(10000, 5, 4800, 4)
# accessHighway(100000, 2, 4800, 4)

