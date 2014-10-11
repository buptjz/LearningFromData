# -*- coding: utf-8 -*-
__author__ = 'wangjz'

"""
Learning From Data
HW 1
Run a computer simulation for flipping 1,000 virtual fair coins. Flip each coin independently 10 times.
Focus on 3 coins as follows: c1 is the first coin flipped, crand is a coin chosen randomly from the 1,000,
and cmin is the coin which had the minimum frequency of heads (pick the earlier one in case of a tie). Let ν1,
νrand, and νmin be the fraction of heads obtained for the 3 respective coins out of the 10 tosses.
Run the experiment 100,000 times in order to get a full distribution of ν1, νrand, and νmin (note that crand
and cmin will change from run to run).
"""

import random


def avg(a_list):
    return 1.0 * sum(a_list) / len(a_list)

TOTAL_COINs = 1000
FLIP_TIMEs = 10
EXP_TIMEs = 100000

coins = [0 for _ in range(TOTAL_COINs)]
minimums = [0 for _ in range(EXP_TIMEs)]
randoms = [0 for _ in range(EXP_TIMEs)]
firsts = [0 for _ in range(EXP_TIMEs)]
for e in range(EXP_TIMEs):
    for c in range(TOTAL_COINs):
        for f in range(FLIP_TIMEs):
            if random.randint(0, 1):
                coins[c] += 1
    minimums[e] = min(coins)
    randoms[e] = coins[random.randint(0, TOTAL_COINs-1)]
    firsts[e] = coins[0]

print(avg(minimums))
print(avg(randoms))
print(avg(firsts))