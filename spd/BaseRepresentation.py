import numpy as np
from .utils import *

class BaseRepresentation:
    """
    Stores a representation of a sum of Paulis or Majoranas or computational basis states, by storing bits in a 2D array (npauli x nbits) and an array of coefficients.
    Bits are packed into 64-bit unsigned integers.
    nq is number of 64-bit unsigned integers needed to store all qubits.
    """
    def __init__(self, bits, coeffs, nq):
        self.bits = bits
        self.coeffs = coeffs
        self.nq = nq

    @property
    def size(self):
        return len(self.bits)

    #Same as in qiskit SparsePauliOp.
    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            key = [key]
        return BaseRepresentation(self.bits[key, :], self.coeffs[key], self.nq)

    def copy(self):
        return BaseRepresentation(self.bits.copy(), self.coeffs.copy(), self.nq)
    def find_element_index(self, other):
        """
        Find index of element (other) or the index where the elements would be added if not existing already in self.
        'other' is also BaseRepresentation and can contain multiple elements (an array of indices is returned).
        """
        return find_bit_index(self.bits, other.bits)
    def find_elements(self, other, index=None):
        """
        Returns logical array indicating whether an element in 'other' is in 'self'.
        If 'index' is provided, it will assume that we already found the indices of elements and only need to compare
        elements from 'other' to elements in 'self' at given indices. 
        """
        if index is None:
            index = self.find_element_index(other)
        return bits_equal_index(self.bits, other.bits, index % self.size)
    def insert_elements(self, other, order=True, serial=True):
        """
        Insert a new element or a list of elements (stored in BaseRepresentation 'other') into 'self'.
        New BaseRepresentation is first ordered by default. Set to False only if you know it is alrady ordered. 
        Can be done in a parallel or serial way.
        """
        if order:
            other.order_elements()
        index = self.find_element_index(other)
        if serial:
            self.bits, self.coeffs = insert_index_serial(self.bits, other.bits, self.coeffs, other.coeffs, index)
        else:
            self.bits, self.coeffs = insert_index(self.bits, other.bits, self.coeffs, other.coeffs, index)
    def delete_elements(self, index, serial):
        """
        Delete elements at indices in array 'index'.
        """
        if serial:
            self.bits, self.coeffs = delete_index_serial(self.bits, self.coeffs, index)
        else:
            self.bits, self.coeffs = delete_index(self.bits, self.coeffs, index)
    def order_elements(self):
        """
        Orders elements in BaseRepresentation by first ordering bits at qubit 1, then bits at qubit 2, and so on.
        """
        indices = np.lexsort([self.bits[:,j] for j in reversed(range(self.bits.shape[1]))])
        self.bits = self.bits[indices]
        self.coeffs = self.coeffs[indices]
    def remove_duplicates(self, serial, order=True, threshold=0):
        """
        Removes duplicate elements in 'self' BaseRepresentation.
        """
        if order:
            self.order_elements()
        remove_duplicates(self.bits, self.coeffs)
        self.delete_elements(np.flatnonzero(abs(self.coeffs)<=threshold), serial=serial)
    def overlap(self, other):
        """
        Computes overlap of two sums as Tr[B^dag A] / N, where N is a normalization factor.
        self (A) and other (B) are both BaseRepresentation objects. 
        """
        index = self.find_element_index(other)
        elements_found = self.find_elements(other, index=index)
        index_found = index[elements_found]
        return np.sum(self.coeffs[index_found] * np.conj(other.coeffs[elements_found]))
    @property
    def weight(self):
        return count_nonzero_array(self.bits)