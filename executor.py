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

accessMerge(10000, 7)
accessMerge(100000, 0)

