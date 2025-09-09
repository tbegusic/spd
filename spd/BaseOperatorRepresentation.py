import numpy as np
from .utils import *
from .BaseRepresentation import BaseRepresentation
from qiskit.quantum_info import SparsePauliOp, PauliList

class BaseOperatorRepresentation(BaseRepresentation):
    '''
    Contains methods that are specific to operators (Pauli or Majorana) and would not make sense for states.
    '''

    @classmethod
    def from_pauli_list(cls, plist, coeffs=None):
        """
        Constructs BaseOperatorRepresentation from qiskit PauliList.
        """
        z = plist._z
        x = plist._x
        bits = np.hstack((packbits(np.array(z)), packbits(np.array(x))))
        c = np.array(coeffs) if coeffs is not None else np.ones(len(bits), dtype=np.complex128)
        return cls(bits, c * (-1j)**(plist._phase%4), int(np.ceil(plist.num_qubits/64)))
    @classmethod
    def from_sparse_pauli_op(cls, op):
        """
        Constructs BaseOperatorRepresentation from qiskit SparsePauliOp.
        """
        plist = op._pauli_list
        coeffs = op.coeffs
        return cls.from_pauli_list(plist, coeffs=coeffs)
    def to_sparse_pauli_op(self, num_qubits):
        """
        Constructs qiskit SparsePauliOp from BaseOperatorRepresentation.
        """
        plist = PauliList([''])
        plist._z = unpackbits(self.bits[:, 0:self.nq], num_qubits)
        plist._x = unpackbits(self.bits[:, self.nq:2*self.nq], num_qubits)
        plist._phase = np.zeros(self.size)
        return SparsePauliOp(data = PauliList(plist), coeffs = self.coeffs, ignore_pauli_phase=False)

    #Same as in qiskit SparsePauliOp.
    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            key = [key]
        return BaseOperatorRepresentation(self.bits[key, :], self.coeffs[key], self.nq)

    def copy(self):
        return BaseOperatorRepresentation(self.bits.copy(), self.coeffs.copy(), self.nq)
    def ztype(self, index=None):
        """
        Returns logical array indicating whether an element in self is composed only of diagonal matrices (no X, no Y for Paulis, only p or identity for Majoranas).
        If integer array 'index' is provided, then the check is performed only at those indices in BaseRepresentation.
        """
        if index is None:
            return np.logical_not(np.any(self.bits[:, self.nq:], axis=1))
        else:
            return np.logical_not(np.any(self.bits[index, self.nq:], axis=1))
    def _count_x(self):
        return count_and(np.bitwise_not(self.bits[:, :self.nq]), self.bits[:, self.nq:])
    def _count_y(self):
        return count_and(self.bits[:, :self.nq], self.bits[:, self.nq:])
    def _count_z(self):
        return count_and(self.bits[:, :self.nq], np.bitwise_not(self.bits[:, self.nq:]))
    def _count_xy(self):
        return count_nonzero_array(self.bits[:, self.nq:])
    @property
    def weight(self):
        return count_or(self.bits[:, :self.nq], self.bits[:, self.nq:])
    @staticmethod
    def exp_val_comp_basis_state(obs, state):
        """
        Computes <psi| O |psi> for observable O and computational basis state |psi>. 
        State is represented by an array of bool variables (False = state 0, True = state 1).
        """
        zs = obs.ztype()
        sign = count_and_array_bool(obs.bits[zs, :obs.nq], packbits(state[np.newaxis, :])[0])
        return np.sum(obs.coeffs[zs]*np.where(sign, -1, 1))
    def apply_to_comp_basis_state(self, state=None):
        """
        Applies operator O to computational basis state |s> to create a sum of computational basis states represented as BaseRepresentation.
        State is provided by a bool array (False = 0, True = 1).
        In general, this can create duplicates in BaseRepresentation which are not automatically removed by this function.
        """
        if state is None:
            state = np.zeros(self.nq, dtype=bool)
        state_bits = packbits(state[np.newaxis, :])
        out_state = BaseRepresentation(self.bits[:,self.nq:].copy(), self.coeffs.copy(), self.nq) 
        inplace_xor(out_state.bits, state_bits[0, :])
        phase = count_and(self.bits[:, :self.nq], out_state.bits)%2
        pmult_sign(out_state.coeffs, 1, phase)
        return out_state
