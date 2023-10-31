import numpy as np
from .utils import *

class PauliRepresentation:
    """
    Stores a representation of a sum or a list of Paulis, by storing z and x bits in a 2D array (npauli x nbits).
    Bits are packed into 64-bit integers.
    Phase is (-i)^p, and we store integer p. 
    nq is number of qubits.
    Coefficients are not used in the initialization but can also be stored inside the class (as done later).
    """
    def __init__(self, bits, phase, nq):
        self.bits = bits
        self.phase = phase
        self.nq = nq

    @staticmethod
    def from_pauli_list(z, x, phase, nq):
        """
        Constructs PauliRepresentation from qiskit PauliList.
        """
        bits = np.hstack((packbits(np.array(z)), packbits(np.array(x))))
        return PauliRepresentation(bits, phase, ceil(nq/64))
    def size(self):
        return len(self.bits)
    def find_pauli_index(self, other):
        """
        Find index of a Pauli (other) or the index where the Pauli would be added if not existing already in self.
        'other' is also PauliRepresentation and can contain multiple Paulis (an array of indices is returned).
        """
        return find_bit_index(self.bits[:,:], other.bits, self.size(), self.nq)
    def find_pauli(self, other, index=None):
        """
        Returns logical array indicating whether a Pauli in 'other' is in 'self'.
        If 'index' is provided, it will assume that we already found the indices of Paulis and only need to compare
        Paulis from 'other' to Paulis in 'self' at given indices. 
        """
        if index is None:
            index = self.find_pauli_index(other)
        return bits_equal(self.bits[index%self.size(), :], other.bits)
    def insert_pauli(self, other, coeffs, serial):
        """
        Insert a new Pauli or a list of Paulis (stored in PauliRepresentation 'other') into 'self'.
        Can be done in a parallel or serial way.
        """
        index = self.find_pauli_index(other)
        if serial:
            self.bits, self.phase, self.coeffs = insert_index_serial(self.bits, other.bits, self.phase, other.phase, self.coeffs, coeffs, index, self.nq)
        else:
            self.bits, self.phase, self.coeffs = insert_index(self.bits, other.bits, self.phase, other.phase, self.coeffs, coeffs, index, self.nq)
    def delete_pauli(self, index, serial):
        """
        Delete Paulis at indices in array 'index'.
        """
        if serial:
            self.bits, self.phase, self.coeffs = delete_index_serial(self.bits, self.phase, self.coeffs, index)
        else:
            self.bits, self.phase, self.coeffs = delete_index(self.bits, self.phase, self.coeffs, index, self.nq)
    def anticommutes(self, other):
        """
        Takes as input PauliRepresentation 'self' (a list of Paulis) and PauliRepresentation 'other' of a single Pauli (!)
        and returns logical array that indicates which Paulis in self anticommute with 'other'.
        """
        a_dot_b = anticommutation_relation(self.bits[:, self.nq:], other.bits[0, :self.nq])
        b_dot_a = anticommutation_relation(self.bits[:, :self.nq], other.bits[0, self.nq:])
        return not_equal(a_dot_b, b_dot_a)
    def compose_with(self, other):
        """
        Composes all Paulis in 'self' with the Pauli (only one Pauli allowed) in 'other'.
        Let A be a Pauli in 'self' and B ='other'. Then the result is A -> B*A (in place multiplication).
        """
        update_phase(self.phase[:], other.phase[0], self.bits[:, :self.nq], other.bits[0, self.nq:])
        inplace_xor(self.bits, other.bits[0, :])
    def order_pauli(self):
        """
        Orders Paulis in PauliRepresentation by first ordering bits at qubit 1, then bits at qubit 2, and so on.
        """
        indices = np.lexsort([self.bits[:,j] for j in reversed(range(2*self.nq))])
        self.bits = self.bits[indices]
        self.phase = self.phase[indices]
        return indices
    def ztype(self, index=None):
        """
        Returns logical array indicating whether a Pauli in self is composed only of Z or identity Pauli matrices (no X, no Y).
        If integer array 'index' is provided, then the check is performed only at those indices in PauliRepresentation.
        """
        if index is None:
            return np.logical_not(np.any(self.bits[:, self.nq:], axis=1))
        else:
            return np.logical_not(np.any(self.bits[index, self.nq:], axis=1))

def evaluate_expectation_value_zero_state(pauli, index):
    """
    Evaluates Pauli expectation value with respect to the |0> state.
    """
    return (-1j)**(pauli.phase[index])