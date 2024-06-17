from datetime import datetime
import os
from tqdm import trange
import time
import math


for idx in trange(20):
    t = time.time()
    seed = int(math.modf(t)[0]*1e7)
    print(seed)
    os.system('python ./envs/generation.py --tag "exc_generation" --env "Exc-v51" --seed {} --gpu "0"'.format(seed))