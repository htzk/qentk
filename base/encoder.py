from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import U1,U2,U3,RandomUnitary
import numpy as np

MAN_HAAR = "manual_haar"
KEYS = [MAN_HAAR]


class NoSuchEncoderException(Exception):
    pass

class QulacsEncoder:
    def __init__(self, nqubit):
        self.nqubit = nqubit

    def encode(self, x) -> QuantumCircuit:
        return QuantumCircuit(self.nqubit)

    def state(self, x) -> QuantumState:
        state = QuantumState(self.nqubit)
        self.encode(x).update_quantum_state(state)
        return state


class QulacsEncoderFactory:
    @classmethod
    def create(cls, key, nqubit) -> QulacsEncoder:
        if key == MAN_HAAR:
            return ManualHaarEncoder(nqubit)
        raise NoSuchEncoderException

class ManualHaarEncoder(QulacsEncoder):
    def __init__(self, nqubit):
        super().__init__(nqubit)

    def encode(self, x, seed):
        np.random.seed(seed=seed)
        theta = np.random.uniform(0, 2*np.pi, self.nqubit*3*4)
        circuit = QuantumCircuit(self.nqubit)
        for i in range(self.nqubit):
            circuit.add_RX_gate(i, x[i])
        for j in range(int(len(theta)/self.nqubit/3)):
            for i in range(self.nqubit):
                circuit.add_U3_gate(i, theta[0 +3*(i+j)], theta[1 +3*(i+j)], theta[2 +3*(i+j)])
                circuit.add_CNOT_gate(i%self.nqubit, (i+1)%self.nqubit)
        return circuit

