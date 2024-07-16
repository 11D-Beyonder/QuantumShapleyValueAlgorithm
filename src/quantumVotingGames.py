import pickle

import matplotlib.pyplot as plt
import numpy as np
import quantumBasicVotingGame as vg
from quantumShapEstimation import QuantumShapleyWrapper as qsw
from tqdm.auto import tqdm

# NOTE: 定义常量

# 模拟次数
numTrials = 128
# 积分的分段数
maxEll = 11

# NOTE: 定义不同的情况

# 玩家数量
numPlayersCond = [2, 4, 8]
# 阈值位数
thresholdBitCond = [3, 4, 5]
# 粗略方差
roughVarianceCond = [1, 1, 2]


# NOTE: 执行一系列模拟，
# 用量子、经典方法计算Shapley值，
# 存储在simulations中。
simulations = {}

# NOTE: 第1层循环：执行numTrials次模拟
for trialNum in tqdm(range(numTrials), desc="Current Trial"):
    # NOTE: 第2层循环：测试不同积分分段数
    for ell in tqdm(range(1, maxEll), desc="Current Ell"):
        # NOTE: 第3层循环：遍历三个条件（玩家数量、阈值位数、粗略方差）。
        for n, thresholdBits, roughVariance in zip(
            numPlayersCond, thresholdBitCond, roughVarianceCond
        ):
            # 一个实验单元trial
            trial = (n, ell, trialNum)
            # 阈值
            threshold = 2 ** (thresholdBits - 1)
            # 生成随机游戏
            playerVals = vg.randomVotingGame(
                numPlayers=n, thresholdBits=thresholdBits, roughVariance=roughVariance
            )

            # 量子计算
            qshaps = vg.quantumVotingShap(
                threshold=threshold, playerVals=playerVals, ell=ell
            )

            # 经典计算
            cshaps = vg.classicalVotingShap(
                threshold=threshold,
                playerVals=playerVals,
            )

            simulations[trial] = (qshaps, cshaps)


with open("shapleyVoteResults.pkl", "wb") as f:
    pickle.dump(simulations, f)


def meanAbsError(qshaps, cshaps):
    err = 0
    for qshap, cshap in zip(qshaps, cshaps):
        err += abs(qshap - cshap)
    return err


plt.rcParams["figure.figsize"] = [12, 5]
fig, ax = plt.subplots(1, len(numPlayersCond))

# We're looking to find reciprocal mean abs error per trial
# For each trial with n players
for i, n in enumerate(numPlayersCond):
    # Orient data
    resultsX = []
    resultsY = []
    resultErr = []
    for ell in range(2, maxEll):
        trialOutcomes = []

        for trialNum in range(numTrials):
            qshaps, cshaps = simulations[(n, ell, trialNum)]
            trialOutcomes.append(meanAbsError(qshaps, cshaps))

        trialOutcomes = np.array(trialOutcomes)
        resultsX.append(ell)
        resultsY.append(trialOutcomes.mean())
        resultErr.append(trialOutcomes.std())

        # resultsX += len(trialOutcomes) * [ell]
        # resultsY += trialOutcomes

    ax[i].set_title(f"{n} Players")  # , Threshold: {2**thresholdBitCond[i]}")
    ax[i].bar(
        np.array(resultsX),
        1 / np.array(resultsY),
        # yerr=resultErr,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax[i].set_xlabel(r"$\ell$")
    ax[i].set_ylabel(r"Reciprocal Mean Absolute Error")
    print(f"{n=}:", 1 / np.array(resultsY))

plt.tight_layout()
plt.show()
