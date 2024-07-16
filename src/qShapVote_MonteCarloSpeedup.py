#!/usr/bin/env python
# coding: utf-8

# ## Quantum Voting Shapley Values and Monte Carlo Speedup
# In this program, we use the methods from Montanaro (2017; Quantum speedup of Monte Carlo methods) to quadratically speed up our quantum algorithm for Shapley value estimation in voting games.
# 
# Additionally, the following tutorial for amplitude estimation was used as a framework: 
# https://qiskit.org/ecosystem/finance/tutorials/00_amplitude_estimation.html 

# #### Importing

# In[121]:


import quantumBasicVotingGame as qvg
from quantumShapEstimation import QuantumShapleyWrapper as qsw

import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit
from qiskit import transpile, Aer
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import QFT


# #### Definitions

# In[122]:


#Debug
debug = True

#Registers
betaApproxBits = 2
numVoters      = 3
numVoteBits    = 3
amplitudeEstBits = 4
targetPlayer   = 1

currentBit = 0

auxReg    = [currentBit+i for i in range(betaApproxBits)]
currentBit += betaApproxBits

playerReg = [currentBit+i for i in range(numVoters)]
currentBit += numVoters

countReg  = [currentBit+i for i in range(numVoteBits)]
currentBit += numVoteBits

#Note utility is just the largest bit of the count Reg
utilReg   = [betaApproxBits+numVoters]

amplitudeReg = [currentBit+i for i in range(amplitudeEstBits)]
currentBit += amplitudeEstBits

print(f"{auxReg=       }")
print(f"{playerReg=    }")
print(f"{countReg=     }")
print(f"{utilReg=      }")
print(f"{amplitudeReg= }")
# print([bit for bit in countReg if bit != utilReg[0]])


# #### Generate Voting Game

# In[123]:


votingPowers = qvg.randomVotingGame(
    numPlayers=numVoters,
    thresholdBits=numVoteBits,
    roughVariance=2,
)

#temp:
votingPowers = [3,2,1]
#:temp

print(f"{votingPowers=}")
print(f"threshold=  {2**(numVoteBits-1)}")


# #### Helper Function

# In[124]:


def assessCircuit(circuit: QuantumCircuit, register: list[int]):
    #Use Aer
    sim = Aer.get_backend('aer_simulator')
    circuit.save_statevector()

    #Compile the circuit down to low-level QASM instructions
    compiled_circuit = transpile(circuit, sim)

    #Simulate circuit
    result = sim.run(compiled_circuit).result()
    out_state = result.get_statevector(circuit, decimals=4)

    #Visualize output
    probs = out_state.probabilities(register)
    state = out_state.to_dict()

    angles = {
        key: np.sin(np.angle(value)/2)
        for key, value in state.items()
    }

    print(state)
    print(angles)
    colors = [f"#{int(180*angle)%256:02x}0080" for angle in angles.values()]
    # print(colors)

    x = np.linspace(0, 1-1/len(probs), len(probs))
    plt.bar(x, probs, align="edge", width=.9/(len(probs)), color=colors)
    plt.xticks(
        ticks=x+.9/(2*len(probs)), 
        labels=[f"{num:0{len(register)}b}" for num in range(len(probs))],
        rotation=90
    )
    plt.show()

    print(probs)

    return probs


# ## Defining $A$ and $Q$ Gates

# The A gate represents the preparation gate which gives each coalition amplitudes which correspond to $\gamma(n,m)$. 
# 
# The Q gate represents the grover operator $Q = AS_0A^{-1}S_\psi$.
# 
# Where $S_0$ is the reflection with respect to $|0>$ and $S_\psi$ is the oracle (vote result).

# #### $A$ and $A^{-1}$ Gates

# In[125]:


# A Gate
ACircuit = QuantumCircuit(len(auxReg + playerReg))

#Prepare auxiliary register
qsw.initBetaApproxBits(auxReg, ACircuit)


ACircuit.append(
    qsw.getShapleyInitGate(
        betaApproxBits=betaApproxBits,
        numFactors=numVoters,
        target=targetPlayer,
        targetOn=False,
    ),
    auxReg + playerReg,
)

print(ACircuit.draw())

AGate = ACircuit.to_gate()
AGate.label = "A"

AInvGate = AGate.inverse()
AInvGate.label = "A^-1"


# #### $S_0$ Gate

# In[126]:


#S0 Gate
S0Circuit = QuantumCircuit(len(set(auxReg+playerReg+countReg+utilReg)))

S0Circuit.x(auxReg+playerReg+[bit for bit in countReg if bit != utilReg[0]])
S0Circuit.mcx(
    control_qubits=auxReg+[bit for bit in countReg if bit != utilReg[0]]+playerReg,
    target_qubit=utilReg
)
S0Circuit.z(utilReg)
S0Circuit.mcx(
    control_qubits=auxReg+[bit for bit in countReg if bit != utilReg[0]]+playerReg,
    target_qubit=utilReg
)
S0Circuit.x(auxReg+playerReg+[bit for bit in countReg if bit != utilReg[0]])
S0Gate = S0Circuit.to_gate(label="S_0")

if debug:
    tempS0Circuit = QuantumCircuit(len(set(auxReg+playerReg+countReg+utilReg)))
    tempS0Circuit.h(playerReg)
    tempS0Circuit += S0Circuit
    assessCircuit(tempS0Circuit, playerReg)

S0Circuit.draw()


# #### $S_\psi$ Gate

# In[127]:


#SPsi Gate
voteOracle, _, _, _ = qvg.randomVotingGameGate(
    thresholdBits=numVoteBits,
    playerVal=votingPowers,
)
voteOracle.label = "vOracle"
voteOracleInverse = voteOracle.inverse()
voteOracleInverse.label = "vOracle^-1"

#This is voting game specific 
def vote(circuit):
    circuit.append(
        voteOracle,
        playerReg + countReg,
    )
    circuit.z(utilReg)
    circuit.append(
        voteOracleInverse,
        playerReg + countReg,
    )

SPsiCircuit = QuantumCircuit(len(set(auxReg+playerReg+countReg+utilReg)))

vote(SPsiCircuit)
SPsiCircuit.x(playerReg[targetPlayer])
vote(SPsiCircuit)
SPsiCircuit.x(playerReg[targetPlayer])

if debug:
    tempSPsiCircuit = QuantumCircuit(len(set(auxReg+playerReg+countReg+utilReg)))
    tempSPsiCircuit.h([player for player in playerReg if player != playerReg[targetPlayer]])
    tempSPsiCircuit += SPsiCircuit
    assessCircuit(tempSPsiCircuit, [player for player in playerReg if player != playerReg[targetPlayer]])

SPsiGate = SPsiCircuit.to_gate()
SPsiGate.label = "S_psi"

SPsiCircuit.draw()


# #### $Q$ Gate

# In[128]:


#Q circuit
QCircuit = QuantumCircuit(len(set(auxReg+playerReg+countReg+utilReg)))

#Phase Gate
phaseCircuit = QuantumCircuit(1)
phaseCircuit.p(np.pi, 0); phaseCircuit.x(0)
phaseCircuit.p(np.pi, 0); phaseCircuit.x(0)
phaseGate = phaseCircuit.to_gate(label="-I")
print(phaseCircuit.draw())

#Main body
QCircuit.append(phaseGate, [0])
QCircuit.append(SPsiGate, auxReg+playerReg+countReg)
QCircuit.append(AInvGate, auxReg+playerReg)
QCircuit.append(S0Gate, auxReg+playerReg+countReg)
QCircuit.append(AGate, auxReg+playerReg)

print(QCircuit.draw())
print(QCircuit.decompose(reps=6).depth())

QGate = QCircuit.to_gate()


# #### $Q$ Circuit Powers

# In[129]:


QGatePowers = []

for i in range(amplitudeEstBits):
    QGatePowers.append(
        QGate.repeat(2**i)
    )
    QGatePowers[i].label = f"Q^{2**i}"
    QGatePowers[i] = QGatePowers[i].control()


# ## Amplitude Estimation Circuit

# In[130]:


ampCircuit = QuantumCircuit(len(set(auxReg+playerReg+countReg+utilReg+amplitudeReg)))

#Preparing the amplitude approximation register
ampCircuit.h(amplitudeReg)

#Preparing the other registers
ampCircuit.append(AGate, auxReg + playerReg)

#Running controlled Q's
for i in range(amplitudeEstBits):
    ampCircuit.append(QGatePowers[i], [amplitudeReg[i]]+auxReg+playerReg+countReg)

#Inverse Fourier transform on the amplitude approx register
qftGate = QFT(num_qubits=amplitudeEstBits, inverse=True).to_gate()
qftGate.label = "QFT^-1"
ampCircuit.append(qftGate, amplitudeReg)

print(ampCircuit.draw())


# ## Running the Circuit

# In[131]:


probs = assessCircuit(ampCircuit, amplitudeReg)


# #### Post-processing Result 

# In[132]:


output = np.sin(
    (np.pi/2**amplitudeEstBits) * np.arange(2**amplitudeEstBits)
)**2
print(output)

estimatedOutput = np.dot(probs, output)
print(estimatedOutput)

