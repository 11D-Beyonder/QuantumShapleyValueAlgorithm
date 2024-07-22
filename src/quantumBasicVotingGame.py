from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate

from quantumShapEstimation import QuantumShapleyWrapper as qsw
from shapExampleGenerator import SHAPGenerator


def constructPlusOneGate(
    numBits: int, numControls: int = 0, name: Optional[str] = None
):
    """
    Gate Structure:
        [Controls] + [Input]

    Returned gate adds one mod 2**numBits to input qubits
    """
    # init circuit
    if name is None:
        name = "+1"
        if numControls > 0:
            name = f"c({numControls}){name}"
    circuit = QuantumCircuit(numBits, name=name)

    # build circuit
    for i in range(numBits - 1):
        mcx = MCXGate(numBits - i - 1)
        circuit.append(mcx, [j for j in range(numBits - 1, i - 1, -1)])
    circuit.x(numBits - 1)

    # convert circuit to gate and return
    if numControls > 0:
        return circuit.to_gate().control(numControls)
    else:
        return circuit.to_gate()


def constructFixedAdditionGate(
    numBits: int, increment: int, numControls: int = 0, name: Optional[str] = None
):
    if name is None:
        name = f"+{increment}"
        if numControls > 0:
            name = f"c({numControls}){name}"

    circuit = QuantumCircuit(numBits, name=name)

    for bit in range(numBits):
        if increment & (1 << bit) != 0:
            gate = constructPlusOneGate(numBits - bit)
            circuit.append(gate, [i for i in range(0, numBits - bit)])

    # print(circuit.draw())
    # convert circuit to gate and return
    if numControls > 0:
        return circuit.to_gate().control(numControls)
    else:
        return circuit.to_gate()


def randomVotingGame(
    numPlayers: int, thresholdBits: int, roughVariance: int = 1
) -> list[int]:
    """分配每个玩家的票数

    Args:
        numPlayers (int): 玩家数量
        thresholdBits (int):  决策通过的票数下限，设定为2^(thresholdBits-1)。
        这样我就可以在设计电路时偷懒，只检查一个位。
        roughVariance (int): 越大表示各玩家的票数差异越大。
    Returns:
        每个玩家的票数列表，举例：[2, 3, 9, 1]。
    """

    # 每个玩家的票数
    playerVotingPower = np.zeros(numPlayers, dtype=int)
    # 分配票数的顺序
    randomOrderPlayers = np.arange(numPlayers)
    np.random.shuffle(randomOrderPlayers)

    threshold = 2 ** (thresholdBits - 1)

    # NOTE: 设置总票数，给每个玩家分配票数。
    totalVotes = int(np.floor((1 + np.random.rand()) * threshold))
    while True:
        for i in randomOrderPlayers:
            while np.random.rand() > 0.5:
                if totalVotes <= 0:
                    break
                # 用roughVariance调控玩家间票数的差距
                variance = int(np.ceil(roughVariance * np.random.rand()))
                change = min(totalVotes, variance)
                playerVotingPower[i] += change
                totalVotes -= change

        if totalVotes <= 0:
            break
        np.random.shuffle(randomOrderPlayers)

    return list(playerVotingPower)


def randomVotingGameGate(thresholdBits: int, playerVal: list[float]):
    playerReg = np.arange(len(playerVal)).tolist()
    voteReg = np.arange(len(playerVal), len(playerVal) + thresholdBits).tolist()
    allReg = playerReg + voteReg
    utilityReg = [len(playerVal)]

    circuit = QuantumCircuit(len(playerReg) + len(voteReg))

    for player in playerReg:
        circuit.append(
            constructFixedAdditionGate(len(voteReg), playerVal[player], 1),
            [player] + voteReg,
        )

    # temp:
    print(circuit.draw())
    #:temp

    return circuit.to_gate(), playerReg, utilityReg, allReg


def classicalVotingShap(threshold: int, playerVals: list[int]) -> list[float]:

    def intToCoalitionValue(coalition: int) -> int:
        """用coalition表示投票情况，将其看作一个二进制数，第j位为1代表第j个玩家投票。
        第j个玩家投票则累加票数，最后统计总票数是否达到阈值，达到则代表价值函数为1。

        Args:
            coalition: 整数，二进制形式表示投票情况。
            高位表示playerVals索引小的玩家。
            举例（从0计数）："0010"，第1位为1，表示playerVals[2]的票数要累加。

        Returns:
            价值函数。
        """
        votes = 0
        for j in range(len(playerVals)):
            jthPlayerOn = (coalition & (1 << j)) > 0
            if jthPlayerOn:
                votes += playerVals[len(playerVals) - j - 1]

        return int(1) if votes >= threshold else int(0)

    coalitionValues = SHAPGenerator.lambdaGenerateContributions(
        numFactors=len(playerVals), generator=intToCoalitionValue
    )

    sg = SHAPGenerator(
        numFactors=len(playerVals),
        rangeMin=int(0),
        rangeMax=int(1),
        contributions=coalitionValues,
    )
    shapleyValues = []
    # 计算每个人的Shapley。
    for i in range(len(playerVals)):
        shapleyValues.append(sg.computeShap(i))

    return shapleyValues


def quantumVotingShap(
    threshold: int,
    playerVals: list[int],
    ell: int = 2,
) -> list[float]:
    """Uses quantum circuit to estimate shapley values of voting game

    Args:
        threshold (int): Number of votes needed for a successful vote.
          Make sure threshold is a power of 2 - since I was lazy with the circuit.
        playerVals (list[int]): The number of votes attributed to each player.

    Returns:
        list[float]: List of Shapley values of players
    """
    thresholdBits = int(np.floor(np.log2(threshold)) + 1)

    gate, playerReg, utilityReg, allReg = randomVotingGameGate(
        thresholdBits=thresholdBits,
        playerVal=playerVals,
    )

    votingQShapWrapper = qsw(
        gate=gate, factorIndices=playerReg, outputIndices=utilityReg, fullIndices=allReg
    )

    numPlayers = len(playerVals)
    qshaps = numPlayers * [0]
    for i in range(numPlayers):
        qshaps[i] = votingQShapWrapper.approxShap(
            i, rangeMin=0, rangeMax=1, betaApproxBits=ell, directProb=True
        )

    return qshaps


def main():
    thresholdBits = 3
    numPlayers = 3

    threshold = 2 ** (thresholdBits - 1)

    rvgArray = randomVotingGame(
        numPlayers=numPlayers,
        thresholdBits=thresholdBits,
        roughVariance=2,
    )

    # quantum Shapley
    qshaps = quantumVotingShap(
        threshold=threshold,
        playerVals=rvgArray,
    )

    # classical Shapley
    cshaps = classicalVotingShap(
        threshold=threshold,
        playerVals=rvgArray,
    )

    print("Player Values:  ", rvgArray)
    print("Threshold:      ", 2 ** (thresholdBits - 1))
    print(80 * "=")
    print("Shapley Values: ")
    for i in range(numPlayers):
        print(f"\tPlayer {i}:")
        print(f"\t\tqshaps[{i}] = {qshaps[i]:.4f}")
        print(f"\t\tcshaps[{i}] = {cshaps[i]:.4f}")

    # plt.hist(rvgArray)
    # plt.show()


if __name__ == "__main__":
    main()
