import numpy as np
from .utils import *
from .BaseOperatorRepresentation import BaseOperatorRepresentation

class PauliRepresentation(BaseOperatorRepresentation):
    """
    Stores a representation of a sum of Paulis, by storing z and x bits in a 2D array (npauli x nbits) and an array of coefficients.
    """
    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            key = [key]
        return PauliRepresentation(self.bits[key, :], self.coeffs[key], self.nq)

    def copy(self):
        return PauliRepresentation(self.bits.copy(), self.coeffs.copy(), self.nq)

    #Alias functions for convenience and back-compatibility.
    def find_pauli_index(self, other):
        """
        Alias for find_element_index of BaseRepresentation.
        """
        return self.find_element_index(other)
    def find_pauli(self, other, index=None):
        """
        Alias for find_elements of BaseRepresentation.
        """
        return self.find_elements(other, index)
    def insert_pauli(self, other, order=True, serial=True):
        """
        Alias for insert_elements of BaseRepresentation.
        """
        self.insert_elements(other, order, serial)
    def delete_pauli(self, index, serial):
        """
        Alias for delete_elements of BaseRepresentation.
        """
        self.delete_elements(index, serial)
    def order_pauli(self):
        """
        Alias for order_elements of BaseRepresentation.
        """
        self.order_elements()

    #Pauli-specific implementations.
    def anticommutes(self, other):
        """
        Takes as input PauliRepresentation 'self' (a list of Paulis) and PauliRepresentation 'other' of a single Pauli (!)
        and returns logical array that indicates which Paulis in self anticommute with 'other'.
        """
        a_dot_b = count_and_array_bool(self.bits[:, self.nq:], other.bits[0, :self.nq])
        b_dot_a = count_and_array_bool(self.bits[:, :self.nq], other.bits[0, self.nq:])
        return (a_dot_b != b_dot_a)
    def compose_with(self, other):
        """
        Composes all Paulis in 'self' with the Pauli (only one Pauli allowed) in 'other'.
        Let A be a Pauli in 'self' and B ='other'. Then the result is A -> B*A (in place multiplication).
        """
        phase = count_and_array_bool(self.bits[:, :self.nq], other.bits[0, self.nq:])
        pmult_sign(self.coeffs, other.coeffs[0], phase)
        inplace_xor(self.bits, other.bits[0, :])
    @property
    def x_count(self):
        return self._count_x() 
    @property
    def y_count(self):
        return self._count_y() 
    @property
    def z_count(self):
        return self._count_z() 
    @property
    def x_weight(self):
        return self._count_xy()