import numpy as np
from copy import deepcopy
from qiskit.quantum_info import *

def apply_clifford_to_Pauli(string, clifford, int_coeff):
    res = string.copy()
    if string.anticommutes(clifford):
        if int_coeff%2 == 1:
            res = string.compose(clifford)
        res._phase -= int_coeff
    return res

def extract_clifford_part(c):
    n = round(c/(np.pi/4))
    c_res = c - n * np.pi/4
    return n%4, c_res

class CliffordSequence:
    """
    Stores a seuqence of Clifford Pauli rotations of the form exp(-i k * (pi/4) * P)
    by keeping the list of Pauli operators P and integer coefficients k.

    Attributes:
    -----------
    ops : PauliList of Pauli strings
    int_coeffs : list of integer coefficients in front of pi/4 for each Pauli string
    """
    def __init__(self, ops = None, int_coeffs = None):
        self.ops = ops
        self.int_coeffs = [] if int_coeffs is None else int_coeffs
    
    def append(self, op, coeff):
        if self.ops is None:
            self.ops = PauliList(op)
        else:
            self.ops = self.ops.insert(self.ops.size, op)
        self.int_coeffs.append(coeff)
    
    def apply_to_Pauli(self, string):
        res = string.copy()
        if self.ops is not None:
            for j in range(self.ops.size):
                res = apply_clifford_to_Pauli(res, self.ops[j], self.int_coeffs[j])
        return res

    def apply_to_PauliList(self, pauli_list):
        res = PauliList(pauli_list).copy()
        for j in range(res.size):
            res[j] = self.apply_to_Pauli(res[j])
        return res

        
class OperatorSequence:
    """
    Stores a seuqence of Pauli rotations of the form exp(-i c * P)
    by keeping the list of Pauli operators P and real coefficients c.
    Stores them as a list of arrays so that each array contains only commuting operators P.

    Attributes:
    -----------
    ops : list of operators split in levels (qiskit PauliList for each level)
    coeffs : coefficients in front of Pauli strings
    clifford_ops : CliffordSequence containing strings and coefficients for each Clifford applied to transform ops
    """
    def __init__(self, ops, coeffs, **kwargs):
        self.ops = [deepcopy(op) for op in ops]
        self.coeffs = np.array([np.array(c, dtype=float) for c in coeffs], dtype=object)
        self.clifford_ops = kwargs.get('clifford_ops', CliffordSequence())
    
    def apply_clifford_up_to_level(self, level, clifford, int_coeff):
        """Applies a Clifford Pauli rotation gate to all Pauli rotations in OperatorSequence up to layer k."""
        for k in range(level):
            for j in range(self.ops[k].size):
                self.ops[k][j] = apply_clifford_to_Pauli(self.ops[k][j], clifford, int_coeff)
        
    def reduce_clifford(self):
        """
        Angle transformation:
        1. Extract Clifford part from coefficient c.
        2. Adjust c -> c - k*pi/4 
        3. If Clifford part is non-zero, apply the Clifford to all preceding Pauli operators.
        """
        for k in range(len(self.ops)):
            for j in range(self.ops[k].size):
                int_coeff, self.coeffs[k][j] = extract_clifford_part(self.coeffs[k][j])
                if int_coeff != 0:
                    clifford = self.ops[k][j].copy()
                    self.apply_clifford_up_to_level(k, clifford, int_coeff)
                    self.clifford_ops.append(clifford, int_coeff)