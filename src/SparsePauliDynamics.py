import numpy as np
from numba import njit, prange, set_num_threads
from math import ceil

from tqdm.notebook import tqdm

powers_of_two = 1 << np.arange(64, dtype=np.uint64)

def my_tqdm(progress_bar, iterable):
    if progress_bar:
        return tqdm(iterable, leave=False)
    else:
        return iterable
    
@njit
def packbits(bool_array):
    ndim1, ndim2 = np.shape(bool_array)
    ndim2_out = ceil(ndim2/64)
    res = np.empty((ndim1, ndim2_out), dtype = np.uint64)
    for i in range(ndim1):
        for j in range(0, ndim2_out):
            tmp = bool_array[i, j*64:min((j+1)*64, ndim2)]
            res[i,j] = np.sum(powers_of_two[0:tmp.size] * tmp)
    return res

@njit
def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count

@njit
def count_nonzero(a):
    s = 0
    for i in range(len(a)):
        s+=countSetBits(a[i])
    return s

@njit(parallel=True)
def count_nonzero_array(a):
    ndim = len(a)
    s = np.zeros(ndim, dtype=np.int32)
    for i in prange(ndim):
        s[i]+=count_nonzero(a[i])
    return s

@njit(parallel=True)
def count_and(a,b):
    c = np.bitwise_and(a, b)
    return count_nonzero_array(c)

@njit(parallel=True)
def not_equal(a,b):
    c = np.empty(len(a), dtype=np.bool_)
    c = (a != b)
    return c

@njit(parallel=True)
def bits_equal(a, b):
    c = np.empty(len(b), dtype=np.bool8)
    for i in prange(len(c)):
        c[i] = np.all(a[i, :] == b[i, :])
    return c

@njit(parallel=True)
def inplace_xor(a,b):
    a[:,:] = np.bitwise_xor(a, b)

@njit(parallel=True)
def a_lt_b(a, b, out):
    for i in prange(len(out)):
        out[i] = np.abs(a[i]) < b

@njit(parallel=True)
def a_gt_b_and_not_c(a, b, c, out):
    for i in prange(len(out)):
        out[i] = (np.abs(a[i]) >= b) & ~c[i]

@njit(parallel=True)
def find_bit_index(a, b, size_a, nq):
    lower = np.repeat(0, len(b))
    upper = np.repeat(size_a, len(b))
    for j in prange(len(b)):
        for i in range(2*nq):
            if upper[j] == lower[j]:
                break
            lower[j] = lower[j] + np.searchsorted(a[lower[j]:upper[j], i], b[j, i], side='left')
            upper[j] = lower[j] + np.searchsorted(a[lower[j]:upper[j], i], b[j, i], side='right')
    return lower

@njit(parallel=True)
def anticommutation_relation(a, b):
    res = np.empty(len(a), dtype=np.int32)
    for i in prange(len(a)):
        res[i] = count_nonzero(np.bitwise_and(a[i,:], b[:]))
    return np.mod(res, 2)

@njit(parallel=True)
def update_phase(p1, p2, a, b):
    for i in prange(len(p1)):
        p1[i] = p1[i] + p2 + 2*count_nonzero(np.bitwise_and(a[i, :], b[:]))
    
@njit(parallel=True)
def insert_index(a,b, ap, bp, ac, bc, index, nq):
    new_size = len(a) + len(b)
    res = np.empty((new_size, 2*nq), dtype=np.uint64)
    res_p = np.empty(new_size, dtype=np.int32)
    res_c = np.empty(new_size, dtype=np.complex128)
    ind = index+np.arange(len(index))
    res[:ind[0]] = a[:index[0]]
    res_p[:ind[0]] = ap[:index[0]]
    res_c[:ind[0]] = ac[:index[0]]
    for i in prange(len(ind)):
        res[ind[i]] = b[i]
        res_p[ind[i]] = bp[i]
        res_c[ind[i]] = bc[i]
        if i==len(ind)-1:
            u = new_size
            ua = len(a)
        else:
            u = ind[i+1]
            ua = index[i+1]
        res[ind[i]+1:u] = a[index[i]:ua]
        res_p[ind[i]+1:u] = ap[index[i]:ua]
        res_c[ind[i]+1:u] = ac[index[i]:ua]
    return res, res_p, res_c

def insert_index_serial(a,b, ap, bp, ac, bc, index, nq):
    new_size = len(a) + len(b)
    res = np.empty((new_size, 2*nq), dtype=np.uint64)
    res_p = np.empty(new_size, dtype=np.int32)
    res_c = np.empty(new_size, dtype=np.complex128)
    mask = np.zeros(new_size, dtype=np.bool_)
    mask[index+np.arange(len(index))] = True
    res[mask] = b[:]
    res_p[mask] = bp[:]
    res_c[mask] = bc[:]
    mask = ~mask
    res[mask] = a[:]
    res_p[mask] = ap[:]
    res_c[mask] = ac[:]
    return res, res_p, res_c

@njit(parallel=True)
def delete_index(a, ap, ac, index, nq):
    new_size = len(a) - len(index)
    res = np.empty((new_size, 2*nq), dtype=np.uint64)
    res_p = np.empty(new_size, dtype=np.int32)
    res_c = np.empty(new_size, dtype=np.complex128)
    mask = np.ones(len(a), dtype=np.bool_)
    mask[index] = False 
    ind = np.nonzero(mask)[0]
    for i in prange(len(ind)):
        res[i] = a[ind[i]]
        res_p[i] = ap[ind[i]]
        res_c[i] = ac[ind[i]]
    return res, res_p, res_c

@njit
def delete_index_serial(a, ap, ac, index):
    mask = np.ones(len(a), dtype=np.bool_)
    mask[index] = False 
    res = a[mask]
    res_p = ap[mask]
    res_c = ac[mask]
    return res, res_p, res_c

@njit(parallel=True)
def pmult(a, b):
    a[:] = a[:] * b

@njit(parallel=True)
def pmult_array(a, b):
    a[:] = a[:] * b[:]

@njit(parallel=True)
def update_coeffs(coeffs1, coeffs2, c, s, p1, p2, index1, index_exists):
    tmp = coeffs2.copy()
    pmult_array(tmp, index_exists * (1j) * s * (-1j)**(p2 - p1))
    coeffs1[index1] = coeffs1[index1] * c + tmp

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

class Simulation:
    """
    Implements sparse Pauli dynamics .

    Attributes:
    -----------
    nq : number of qubits
    observable : Heisenberg-evolved observable represented as a sum of Paulis (PauliRepresentation class)
    operator_sequence : Sequence of Pauli rotation gates (OperatorSequence class)
    depth : number of layers, each composed of commuting Pauli rotation gates
    threshold : parameter for truncating the representation of the observable
    sin_coeffs, cos_coeffs : stores precomputed sine and cosine of rotation angles, which are used repeatedly
    eval_exp_val : A general function for evaluating the expectation value of a Pauli
    nprocs : number of processes for parallel runs (default value: 1)
    """
    def __init__(self, observable, operator_sequence, depth, **kwargs):
        self.nq = observable.num_qubits
        self.observable = PauliRepresentation.from_pauli_list(observable._z, observable._x, observable._phase, self.nq)
        self.observable.coeffs = np.array(kwargs.get('observable_coeffs', [1.0]), dtype=np.complex128)
        self.observable.coeffs = self.observable.coeffs[self.observable.order_pauli()]
        self.operator_sequence = operator_sequence
        self.depth = depth
        self.threshold = kwargs.get('threshold', 0.01)
        coeffs = [np.array([c for c in clist]) for clist in operator_sequence.coeffs]
        self.sin_coeffs = [np.sin(2*c) for c in coeffs]
        self.cos_coeffs = [np.cos(2*c) for c in coeffs]
        self.eval_exp_val = kwargs.get('exp_val_fun', None)
        self.nprocs = kwargs.get('nprocs', 1)
        set_num_threads(self.nprocs)

        if self.eval_exp_val is None:
            self.eval_exp_val = evaluate_expectation_value_zero_state
    
    def run(self):
        """
        Runs the simulation:
        Loops through all gates and applies each gate. At the end, computes the expectation value and returns it.
        """
        for k in range(self.depth):
            for j in range(len(self.operator_sequence.ops[k])):
                op = PauliRepresentation.from_pauli_list(self.operator_sequence.ops[k][j]._z, self.operator_sequence.ops[k][j]._x, self.operator_sequence.ops[k][j]._phase, self.nq)
                self.apply_gate(k, j, op)
        nonzero_pauli_indices = np.where(self.observable.ztype())[0]
        return np.sum(self.observable.coeffs[nonzero_pauli_indices] * self.eval_exp_val(self.observable, nonzero_pauli_indices))

    def apply_gate(self, k, j, op):
        """
        Applies a gate j in layer k, defined by Pauli operator op.
        The steps are:
        1. Identify set of Paulis in self.observable that anticommute with op.
        2. Compute new Paulis and update coefficients of existing Paulis.
        3. Discard existing Paulis from self.observable whose coefficients are below the threshold.
        4. Insert new Paulis from new_paulis into self.observable where coefficients of new Paulis are above the threshold. 
        """
        anticommuting = np.where(self.observable.anticommutes(op))[0]
        if len(anticommuting):

            #Find subset of Pauli operators that anticommute with the gate Pauli
            #and compute new Paulis obtained by composition with gate Pauli.
            new_paulis, new_pauli_indices, new_pauli_in_observable = self.prepare_new_paulis(self.observable, anticommuting, op)

            #Update coefficients for existing Paulis and precompute sin(theta) for those that will be added later.
            coeffs_sin = np.array(self.observable.coeffs[anticommuting])
            pmult(coeffs_sin, (1j) * self.sin_coeffs[k][j])
            update_coeffs(self.observable.coeffs, self.observable.coeffs[new_pauli_indices%self.observable.size()], self.cos_coeffs[k][j], self.sin_coeffs[k][j], new_paulis.phase, self.observable.phase[new_pauli_indices%self.observable.size()], anticommuting, new_pauli_in_observable)

            #Project out Paulis and their coefficients that are below threshold.
            to_add_remove = np.empty(len(anticommuting), dtype=np.bool_)
            a_lt_b(self.observable.coeffs[anticommuting], self.threshold, to_add_remove)
            if np.any(to_add_remove):
                self.observable.delete_pauli(anticommuting[to_add_remove], self.nprocs==1)

            #Find which Paulis will be added to the observable.
            to_add_remove = (np.abs(coeffs_sin) >= self.threshold) & np.logical_not(new_pauli_in_observable)
            a_gt_b_and_not_c(coeffs_sin, self.threshold, new_pauli_in_observable, to_add_remove)
            if np.any(to_add_remove):
                self.add_new_paulis(new_paulis, coeffs_sin, to_add_remove)
    
    def prepare_new_paulis(self, obs, anticommuting_ind, op):
        """
        Obtain new Pauli operators by mutliplying self.observable by op.
        Find indices of new Paulis in self.observable (new_pauli_indices) and check if they exist in self.observable already (new_pauli_in_observable logical array).
        """
        new_paulis = PauliRepresentation(obs.bits[anticommuting_ind, :], obs.phase[anticommuting_ind], obs.nq)
        new_paulis.compose_with(op)
        new_pauli_indices = obs.find_pauli_index(new_paulis)
        new_pauli_in_observable = obs.find_pauli(new_paulis, new_pauli_indices)
        return new_paulis, new_pauli_indices, new_pauli_in_observable

    def add_new_paulis(self, new_paulis, new_coeffs, ind_to_add):
        """
        Add rows of new_paulis at indices ind_to_add (these include Paulis that are above threshold and don't exist already in self.observable)
        to self.observable.
        """
        paulis_to_add = PauliRepresentation(new_paulis.bits[ind_to_add, :], new_paulis.phase[ind_to_add], new_paulis.nq)

        #Order Paulis before we add them so that they are correctly inserted simultaneously.
        #This one line orders Paulis and uses reordered indices to order coefficients.
        coeffs_to_add = new_coeffs[ind_to_add][paulis_to_add.order_pauli()]

        #Insert new Paulis and return new array of coefficients.
        self.observable.insert_pauli(paulis_to_add, coeffs_to_add, self.nprocs==1)

    @staticmethod
    def evolve_pauli_sum(pauli_list, operator_sequence, depth, coeffs = None, progress_bar = True, **kwargs):
        """
        A wrapper for evaluating an expectation value of a sum of Paulis by first splitting the pauli_list into individual Paulis
        and constructing and running Simulation for each Pauli.
        """
        res = []
        for i, pauli in enumerate(my_tqdm(progress_bar, pauli_list)):
            c = np.array([1.0]) if coeffs is None else np.array(coeffs[i])
            res.append(Simulation(pauli, operator_sequence=operator_sequence, depth=depth, observable_coeffs=c, **kwargs).run())
        return sum([r for r in res])
