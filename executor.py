from trainHiMeEnvDQN import main as HiMe
from trainMergeEnvDQN import main as Merge
from accessModelHighway import main as accessHighway
from accessModelMerge import main as accessMerge
from trainHighwayEnvCriticalsDQN import main as trainCritHi
from trainHighwayEnvDQN import main as trainHi
import time, numpy


# start = time.time()
# trainHi(1000, 5)
# trainCritHi()
# crit = time.time() - start

# start = time.time()
# trainHi(100000, 0)
# simple = time.time() - start

accessHighway(10000, 7, 300, 0)
accessHighway(100000, 0, 300, 0)
accessHighway(10000, 7, 1500, 1)
accessHighway(100000, 0, 1500, 1)
accessHighway(10000, 7, 2600, 2)
accessHighway(100000, 0, 2600, 2)
accessHighway(10000, 7, 3700, 3)
accessHighway(100000, 0, 3700, 3)
accessHighway(10000, 7, 4800, 4)
accessHighway(100000, 0, 4800, 4)

