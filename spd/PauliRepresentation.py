import numpy as np
from .utils import *
from qiskit.quantum_info import SparsePauliOp, PauliList

class PauliRepresentation:
    """
    Stores a representation of a sum or a list of Paulis, by storing z and x bits in a 2D array (npauli x nbits).
    Bits are packed into 64-bit integers.
    Phase is (-i)^p, and we store integer p. 
    nq is number of qubits.
    Coefficients are not used in the initialization but can also be stored inside the class (as done later).
    """
    def __init__(self, bits, phase, nq, coeffs=None):
        self.bits = bits
        self.phase = phase
        self.nq = nq
        self.coeffs = coeffs

    @staticmethod
    def from_pauli_list(plist, coeffs=None):
        """
        Constructs PauliRepresentation from qiskit PauliList.
        """
        z = plist._z
        x = plist._x
        phase = plist._phase
        nq = plist.num_qubits
        bits = np.hstack((packbits(np.array(z)), packbits(np.array(x))))
        return PauliRepresentation(bits, phase, ceil(nq/64), coeffs=coeffs)
    @staticmethod
    def from_sparse_pauli_op(op):
        """
        Constructs PauliRepresentation from qiskit SparsePauliOp.
        """
        plist = op._pauli_list.copy()
        coeffs = op.coeffs.copy()
        return PauliRepresentation.from_pauli_list(plist, coeffs=coeffs)
    def to_sparse_pauli_op(self, num_qubits):
        """
        Constructs qiskit SparsePauliOp from PauliRepresentation.
        """
        plist = PauliList.from_symplectic(unpackbits(self.bits[:, 0:self.nq], num_qubits), unpackbits(self.bits[:, self.nq:2*self.nq], num_qubits), self.phase % 4)
        return SparsePauliOp(data = plist, coeffs = self.coeffs, ignore_pauli_phase=True)
    def size(self):
        return len(self.bits)
    def copy(self):
        return PauliRepresentation(self.bits.copy(), self.phase.copy(), self.nq, coeffs=self.coeffs.copy())
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
        return bits_equal_index(self.bits, other.bits, index % self.size())
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
    def anticommutes_list(self, other):
        """
        Takes as input PauliRepresentation 'self' (a list of Paulis) and PauliRepresentation 'other'
        and returns 2D logical array that indicates which Paulis in 'self' anticommute with the Paulis in 'other'.
        """
        a_dot_b = anticommutation_relation_list(self.bits[:, self.nq:], other.bits[:, :self.nq])
        b_dot_a = anticommutation_relation_list(self.bits[:, :self.nq], other.bits[:, self.nq:])
        return not_equal(a_dot_b, b_dot_a)
    def compose_with(self, other):
        """
        Composes all Paulis in 'self' with the Pauli (only one Pauli allowed) in 'other'.
        Let A be a Pauli in 'self' and B ='other'. Then the result is A -> B*A (in place multiplication).
        """
        update_phase(self.phase[:], other.phase[0], self.bits[:, :self.nq], other.bits[0, self.nq:])
        inplace_xor(self.bits, other.bits[0, :])
    def compose(self, other, mask):
        """
        Composes all Paulis in 'self' with all Paulis in 'other' for each pair where 'mask' is True.
        Let A be a Pauli in 'self' and B ='other'. Then the result is C = B*A.
        """
        c, cp, cc = compose_mask(self.bits, self.phase, self.coeffs, other.bits, other.phase, other.coeffs, mask, np.count_nonzero(mask, axis=0), self.nq)
        return PauliRepresentation(c, cp, self.nq, coeffs=cc)
    def order_pauli(self):
        """
        Orders Paulis in PauliRepresentation by first ordering bits at qubit 1, then bits at qubit 2, and so on.
        """
        indices = np.lexsort([self.bits[:,j] for j in reversed(range(2*self.nq))])
        self.bits = self.bits[indices]
        self.phase = self.phase[indices]
        return indices
    def remove_duplicates(self, serial, order=True, threshold=0):
        if order:
            self.coeffs = self.coeffs[self.order_pauli()]
        remove_duplicates(self.bits, self.phase, self.coeffs)
        self.delete_pauli(np.flatnonzero(abs(self.coeffs)<=threshold), serial=serial)
    def overlap(self, other):
        """
        Computes overlap of two Pauli sums as Tr[B^dag A] / N, where N is a normalization factor.
        self (A) and other (B) are both PauliRepresentation objects. 
        """
        index = self.find_pauli_index(other)
        pauli_found = self.find_pauli(other, index=index)
        index_found = index[pauli_found]
        return np.sum(self.coeffs[index_found] * np.conj(other.coeffs[pauli_found]) * (-1j)**(self.phase[index_found] - other.phase[pauli_found]))
    def ztype(self, index=None):
        """
        Returns logical array indicating whether a Pauli in self is composed only of Z or identity Pauli matrices (no X, no Y).
        If integer array 'index' is provided, then the check is performed only at those indices in PauliRepresentation.
        """
        if index is None:
            return np.logical_not(np.any(self.bits[:, self.nq:], axis=1))
        else:
            return np.logical_not(np.any(self.bits[index, self.nq:], axis=1))
    def apply_h(self, h):
        """
        Computes [H,A], where A is Pauli representation 'self' and H is '0.5*h' (h is another Pauli representation, typically a Hamiltonian).
        """
        anticommuting = self.anticommutes_list(h)
        if (np.any(anticommuting)):
            out = self.compose(h, anticommuting)
        else:
            out = None 
        return out
    def sum_with_threshold(self, other, threshold, serial):
        if other is not None:
            index = self.find_pauli_index(other) % self.size()
            found = self.find_pauli(other, index)
            tmp = tmp_product(other.coeffs, other.phase, self.phase, index, found)
            add_to_array(self.coeffs, tmp, index)

            to_remove = np.empty(self.size(), dtype=np.bool_)
            a_lt_b(self.coeffs, threshold, to_remove)
            if np.any(to_remove):
                self.delete_pauli(np.flatnonzero(to_remove), serial)

            to_add = np.empty(other.size(), dtype=np.bool_)
            a_gt_b_and_not_c(other.coeffs, threshold, found, to_add)
            if np.any(to_add):
                paulis_to_add = PauliRepresentation(other.bits[to_add, :], other.phase[to_add], other.nq, coeffs=other.coeffs[to_add])
                paulis_to_add.remove_duplicates(serial)
                self.insert_pauli(paulis_to_add, paulis_to_add.coeffs, serial)

def evaluate_expectation_value_zero_state(pauli, index):
    """
    Evaluates Pauli expectation value with respect to the |0> state.
    """
    return (-1j)**(pauli.phase[index])
