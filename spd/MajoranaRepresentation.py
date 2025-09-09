import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .utils import *
from .BaseOperatorRepresentation import BaseOperatorRepresentation

class MajoranaRepresentation(BaseOperatorRepresentation):
    """
    Stores a representation of a sum of Majoranas, by storing c and p bits in a 2D array (nmajorana x nbits) and an array of coefficients.
    """
    @staticmethod
    def fermionic_to_sparse_pauli_op(op):
        """Constructs SparsePauliOp from FermionicOp so that the new object represents Majorana strings (i.e., not a mapping to qubits).
           This then allows us to use the tools from BaseOperatorRepresentation.
        """
        res = []
        for i in op.index_order().terms():
            s=SparsePauliOp.from_sparse_list([('', [], i[1])], num_qubits=op.register_length)
            for j in i[0]:
                s = s.dot(SparsePauliOp.from_sparse_list([('X', [j[1]], 0.5), ('Y', [j[1]], -0.5j if j[0]=='+' else 0.5j)], num_qubits=op.register_length))
            res.append(s.simplify())
        return sum(res).simplify()

    @classmethod
    def from_fermionic_op(cls, op):
        """Initializes MajoranaRepresentation from Qiskit FermionicOp."""
        spo = cls.fermionic_to_sparse_pauli_op(op)
        return cls.from_sparse_pauli_op(spo)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            key = [key]
        return MajoranaRepresentation(self.bits[key, :], self.coeffs[key], self.nq)

    def copy(self):
        return MajoranaRepresentation(self.bits.copy(), self.coeffs.copy(), self.nq)

    #Alias functions for convenience and back-compatibility.
    def find_majorana_index(self, other):
        """
        Alias for find_element_index of BaseRepresentation.
        """
        return self.find_element_index(other)
    def find_majorana(self, other, index=None):
        """
        Alias for find_elements of BaseRepresentation.
        """
        return self.find_elements(other, index)
    def insert_majorana(self, other, order=True, serial=True):
        """
        Alias for insert_elements of BaseRepresentation.
        """
        self.insert_elements(other, order, serial)
    def delete_majorana(self, index, serial):
        """
        Alias for delete_elements of BaseRepresentation.
        """
        self.delete_elements(index, serial)
    def order_majorana(self):
        """
        Alias for order_elements of BaseRepresentation.
        """
        self.order_elements()

    #Majorana-specific implementations.
    def anticommutes(self, other):
        """
        Takes as input MajoranaRepresentation 'self' (a list of Majoranas) and MajoranaRepresentation 'other' of a single Majorana (!)
        and returns logical array that indicates which Majoranas in self anticommute with 'other'.
        """
        r = count_and_array_bool(self.bits, np.concatenate([other.bits[0,self.nq:], np.bitwise_xor(other.bits[0,:self.nq], other.bits[0, self.nq:])]))
        if parity(other.bits[:, self.nq:])[0]:
            r^= parity(self.bits[:, self.nq:])
        return r
    def compose_with(self, other):
        """
        Composes all Majoranas in 'self' with the majorana (only one Majorana allowed) in 'other'.
        Let A be a Majorana in 'self' and B ='other'. Then the result is A -> B*A (in place multiplication).
        """
        phase = count_and_array_bool(self.bits, np.concatenate([other.bits[0, self.nq:], parity_repr(other.bits[0, self.nq:])]))
        if parity(other.bits[:, self.nq:])[0]:
            phase^=parity(self.bits[:, self.nq:])
        pmult_sign(self.coeffs, other.coeffs[0], phase)
        inplace_xor(self.bits, other.bits[0, :])
    @property
    def c_count(self):
        """Counts number of c Majoranas."""
        return self._count_x()
    @property
    def ct_count(self):
        """Counts number of \tilde{c} Majoranas."""
        return self._count_y()
    @property
    def p_count(self):
        "Counts number of p = i\tilde{c}c Majoranas."
        return self._count_z()
    @property
    def nonp_count(self):
        """Counts number of non-p Majoranas."""
        return self._count_xy()