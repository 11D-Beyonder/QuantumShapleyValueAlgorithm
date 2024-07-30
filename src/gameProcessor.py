from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate

from quantumShapEstimation import QuantumShapleyWrapper
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

    # NOTE: 电路形如
    #      ┌───┐
    # q_0: ┤ X ├──────────
    #      └─┬─┘┌───┐
    # q_1: ──■──┤ X ├─────
    #        │  └─┬─┘┌───┐
    # q_2: ──■────■──┤ X ├
    #                └───┘
    # 当所有的MCXGate和最后的X门结合起来时，
    # 它们共同实现了一个量子比特串的加一操作。
    # 如果量子比特串的初始状态表示一个数值，
    # 应用这个量子电路后，数值会增加1。
    # 举例：
    # - |111⟩ -> |000⟩
    # - |101⟩ -> |110⟩

    for i in range(numBits - 1):
        mcx: MCXGate = MCXGate(numBits - i - 1)
        circuit.append(mcx, [j for j in range(numBits - 1, i - 1, -1)])
    circuit.x(numBits - 1)

    if numControls > 0:
        return circuit.to_gate().control(numControls)
    else:
        return circuit.to_gate()


def constructFixedAdditionGate(
    numBits: int, increment: int, numControls: int = 0, name: Optional[str] = None
):
    """构造增加票数的门。

    Args:
        numBits: 票数寄存器的比特数。
        increment: 需要增加的票数。
        numControls: 控制位的比特数。
        name: 门的名称。

    Returns:
        若numControls不为0,则返回一个受控门，否则返回一个普通的门。
    """
    if name is None:
        name = f"+{increment}"
        if numControls > 0:
            name = f"c({numControls}){name}"

    circuit = QuantumCircuit(numBits, name=name)

    # NOTE: increment某一位为1,则在该位上构造一个加1的门。
    # numBits-bit 表示从哪里开始考虑进位。
    # ---
    # 这里其实是一步优化，最朴素的做法是做
    # `increment`次`constructPlusOneGate(numBits=numBits)`
    for bit in range(numBits):
        if increment & (1 << bit) != 0:
            gate = constructPlusOneGate(numBits - bit)
            circuit.append(gate, [i for i in range(numBits - bit)])

    if numControls > 0:
        return circuit.to_gate().control(numControls)
    else:
        return circuit.to_gate()


def randomVotingGameGate(thresholdBits: int, playerVals: list[int]):
    """构造加票数的门，每个玩家作为一个控制比特。
    q_0: ─────■────────────────
              │
    q_1: ─────┼──────────■─────
         ┌────┴────┐┌────┴────┐
    q_2: ┤0        ├┤0        ├
         │         ││         │
    q_3: ┤1 c(1)+1 ├┤1 c(1)+3 ├
         │         ││         │
    q_4: ┤2        ├┤2        ├
         └─────────┘└─────────┘
    Args:
        thresholdBits: 表示票数下限的比特数。
        playerVal: 各玩家的票数。

    Returns:
        加票数的门，玩家寄存器（Pl），票数寄存器（Aux）。
    """
    # 文中Pl寄存器
    playerReg: list[int] = np.arange(len(playerVals)).tolist()
    # 文中Aux寄存器
    voteReg = np.arange(len(playerVals), len(playerVals) + thresholdBits).tolist()
    allReg = playerReg + voteReg
    # HACK: 文中的辅助比特寄存器，不应该是len(playerVal)+thresholdBits+1吗？
    utilityReg = [len(playerVals)]
    circuit = QuantumCircuit(len(playerReg) + len(voteReg))

    # NOTE: 构造加投票的受控门：
    # 玩家比特为控制位，Aux寄存器中比特可以加上对应的票数。

    for player in playerReg:
        circuit.append(
            constructFixedAdditionGate(len(voteReg), playerVals[player], 1),
            [player] + voteReg,
        )

    return circuit.to_gate(), playerReg, utilityReg, allReg


def generateRandomGame(
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
    """使用量子算法计算每个玩家的夏普利值。

    Args:
        threshold (int): 决策执行的票数下限，为了简化这里必须保证是2的幂。
        playerVals (list[int]): 每个玩家的票数。
        ell (int): 积分的分段数。

    Returns:
        list[float]: 每个玩家的夏普利值。
    """

    # NOTE: 最高位为1则代表达到阈值
    thresholdBits = int(np.floor(np.log2(threshold)) + 1)

    gate, playerReg, utilityReg, allReg = randomVotingGameGate(
        thresholdBits=thresholdBits,
        playerVals=playerVals,
    )

    votingQShapWrapper = QuantumShapleyWrapper(
        gate=gate, factorIndices=playerReg, outputIndices=utilityReg, fullIndices=allReg
    )

    qshaps = []
    for i in range(len(playerVals)):
        qshaps.append(
            votingQShapWrapper.approxShap(
                i, rangeMin=0, rangeMax=1, betaApproxBits=ell, directProb=True
            )
        )

    return qshaps


def main():
    thresholdBits = 3
    numPlayers = 3

    threshold = 2 ** (thresholdBits - 1)

    rvgArray = generateRandomGame(
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
