from trainHighwayEnvDQN import main as Highway
from trainHiMeEnvDQN import main as HiMe
from trainMergeEnvDQN import main as Merge
from accessModelHighway import main as accessHighway
from accessModelMerge import main as accessMerge
from trainHighwayEnvCriticalsDQN import main as trainCritHi
import time, numpy


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
# accessHighway(100000, 1, 300, 0)
accessHighway(10000, 6, 3000, 1)
# accessHighway(100000, 1, 1500, 1)
accessHighway(10000, 6, 30000, 2)
#accessHighway(100000, 1, 2600, 2)
accessHighway(10000, 6, 300000, 3)
#accessHighway(100000, 1, 3700, 3)
accessHighway(10000, 6, 3000000, 4)
#accessHighway(100000, 1, 4800, 4)

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

