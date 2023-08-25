from OperatorSequence import OperatorSequence
import numpy as np
from qiskit.quantum_info import *

#Procedures for building IBM's Heavy Hexagon coupling map.
def build_row(start, end):
    return [[i, i+1] for i in range(start, end)]
def build_bridges(start_ind_row, start_ind_bridge):
    return [[i, j] for i, j in zip(range(start_ind_row, start_ind_row+13, 4), range(start_ind_bridge, start_ind_bridge + 4))]

def coupling_list():
    row_indices = [(0,13), (18,32), (37,51), (56,70), (75,89),(94,108),(113,126)]
    bridge_indices = [(0,14), (18,14), (20,33), (39,33), (37,52), (56,52), (58,71), (77,71), (75,90), (94,90), (96,109),(114,109)]
    z=[]
    for r in row_indices:
        z += build_row(*r)
    for b in bridge_indices:
        z +=  build_bridges(*b)
    return z

#Construct a nq-qubit Pauli operator with op=X,Y,Z at index ind.
def pauli_op(ind, nq, op):
    return Pauli(''.join([op if i in ind else 'I' for i in range(nq)]))
#Construct a PauliList of ZZ or X operators.
def ZZ_list(adj_list, nq):
    return PauliList([pauli_op(e, nq,'Z') for e in adj_list])
def X_list(nq):
    return PauliList([pauli_op([i], nq, 'X') for i in range(nq)])

#Construct a OperatorSequence for n layers, given the coupling map (adj_list) and angle theta for the X rotations.
def op_seq_layers(n_layers, adj_list, theta, nq, n_zz):
    ZZ_ops = ZZ_list(adj_list, nq)
    X_ops = X_list(nq)
    ops = []
    coeffs = []
    for _ in range(n_layers):
        ops += [ZZ_ops.copy()] + [X_ops.copy()]
        coeffs += [np.array([-np.pi/4]*n_zz)] + [np.array([theta]*nq)]
    return OperatorSequence(ops, coeffs)

#Construct Pauli operator with a combination of X, Y, Z in ind=[ind_x, ind_y, ind_z].
def pauli_op_XYZ(ind, nq):
    return pauli_op(ind[0], nq, 'X') & pauli_op(ind[1], nq, 'Y') & pauli_op(ind[2], nq, 'Z')

#In the IBM simulation, every other gate will be Clifford, so their coefficients after extracting the Clifford part will be zero and we don't need to simulate those gates.
def extract_non_clifford(op_seq):
    return op_seq.ops[1::2]

#Same as above but modified for the results of fig 3d of arXiv:2308.05077.
def op_seq_layers_3d(n_layers, adj_list, theta, nq, n_zz):
    ZZ_ops = ZZ_list(adj_list, nq)
    X_ops = X_list(nq)
    ops = [X_ops.copy()]
    coeffs = [np.array([theta]*nq)]
    for l in range(n_layers):
        ops += [ZZ_ops.copy()] + [X_ops.copy()]
        coeffs += [np.array([-np.pi/4]*n_zz)] + [np.array([theta]*nq)]
    return OperatorSequence(ops, coeffs)

def extract_non_clifford_3d(op_seq):
    return op_seq.ops[0::2]