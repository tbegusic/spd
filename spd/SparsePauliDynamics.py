import numpy as np
from .utils import *
from .PauliRepresentation import PauliRepresentation
from .MajoranaRepresentation import MajoranaRepresentation
from qiskit.quantum_info import SparsePauliOp

class Simulation:
    """
    Implements sparse Pauli dynamics .

    Attributes:
    -----------
    observable : Heisenberg-evolved observable represented as a sum of Paulis (PauliRepresentation class)
    operator_sequence : Sequence of Pauli rotation gates (SparsePauliOp class)
    sin_coeffs, cos_coeffs : stores precomputed sine and cosine of rotation angles, which are used repeatedly
    threshold : parameter for truncating the representation of the observable
    nprocs : number of processes for parallel runs (default value: 1)
    eval_exp_val : A general function for evaluating the expectation value of a Pauli
    """
    def __init__(self, observable, operator_sequence, **kwargs):

        #Observable.
        self.observable = observable
        self.observable.order_elements()

        self.init_operator_sequence(observable, operator_sequence)

        #Threshold. Default value is 0.01 to ensure that one does not accidentally run a demanding calculation.
        self.threshold = kwargs.get('threshold', 0.01)

        #Number of processors used.
        self.nprocs = kwargs.get('nprocs', 1)
        set_num_threads(self.nprocs)

        #Method to evaluate expectation value of Pauli over the state. By default returns 1.
        self.eval_exp_val = kwargs.get('exp_val_fun', None)
        if self.eval_exp_val is None:
            self.eval_exp_val = lambda _a, _b: 1

    def init_operator_sequence(self, observable, operator_sequence):
        """
        Prepares coefficients for the rotation operators.
        Takes into account that the operator should be Hermitian before computing sin and cos of angle.
        """
        if isinstance(observable, PauliRepresentation):
            self.operator_sequence = PauliRepresentation.from_sparse_pauli_op(operator_sequence)
            phase = self.operator_sequence.y_count%2
        elif isinstance(observable, MajoranaRepresentation):
            self.operator_sequence = MajoranaRepresentation.from_sparse_pauli_op(operator_sequence)
            phase = self.operator_sequence.ct_count%2 + (self.operator_sequence.nonp_count%4)//2
        self.operator_sequence.coeffs *= (-1j)**phase
        self.sin_coeffs = np.sin(2*self.operator_sequence.coeffs)
        self.cos_coeffs = np.cos(2*self.operator_sequence.coeffs)
        self.operator_sequence.coeffs = 1j * (1j)**phase * self.sin_coeffs 

    @classmethod
    def from_pauli_list(cls, observable, operator_sequence, **kwargs):
        """
        Initializes Simulation from observable given as SparsePauliOp or PauliList. Other keyword arguments are simply passed to __init__.
        """
        if isinstance(observable, SparsePauliOp):
            return cls(PauliRepresentation.from_sparse_pauli_op(observable), operator_sequence, **kwargs)
        else:
            return cls(PauliRepresentation.from_pauli_list(observable, coeffs=kwargs.get('observable_coeffs')), operator_sequence, **kwargs)
    
    def run(self):
        """
        Runs SPD simulation.
        Loops through all gates and applies each gate.
        """
        for j in range(self.operator_sequence.size):
            self.apply_gate(j, self.operator_sequence[j])

    def run_circuit(self):
        """
        Runs a circuit simulation:
        First performs SPD propagation and then computes the expectation value and returns it.
        """
        self.run()
        nonzero_pauli_indices = np.where(self.observable.ztype())[0]
        return np.sum(self.observable.coeffs[nonzero_pauli_indices] * self.eval_exp_val(self.observable, nonzero_pauli_indices))

    def run_dynamics(self, nsteps, process = None, process_every = 1, td_ham = None):
        r = []
        if process is not None:
            r.append(process(self.observable))
        for step in range(nsteps):
            if td_ham is not None:
                self.init_operator_sequence(self.observable, td_ham(step))
            self.run()
            if process is not None and ((step+1) % process_every == 0):
                r.append(process(self.observable))
        if process is not None:
            return r

    def apply_gate(self, j, op):
        """
        Applies a gate j, defined by Pauli operator op.
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
            new_paulis, existing_indices, new_pauli_in_observable = self.prepare_new_paulis(self.observable, anticommuting, op)

            #Update coefficients for existing Paulis.
            pmult_mask(self.observable.coeffs, self.cos_coeffs[j], anticommuting)
            psum_index(self.observable.coeffs, new_paulis.coeffs[new_pauli_in_observable], existing_indices)

            #Project out Paulis and their coefficients that are below threshold.
            to_add_remove = np.empty(len(anticommuting), dtype=np.bool_)
            a_lt_b(self.observable.coeffs[anticommuting], self.threshold, to_add_remove)
            if np.any(to_add_remove):
                self.observable.delete_elements(anticommuting[to_add_remove], serial=(self.nprocs==1))

            #Find which Paulis will be added to the observable.
            a_gt_b_and_not_c(new_paulis.coeffs, self.threshold, new_pauli_in_observable, to_add_remove)
            if np.any(to_add_remove):
                #Before inserting, new paulis must be sorted so that they can be correctly inserted simultaneously.
                self.observable.insert_elements(new_paulis[to_add_remove], order=True, serial=(self.nprocs==1))
    
    def prepare_new_paulis(self, obs, anticommuting_ind, op):
        """
        Obtain new Pauli operators by mutliplying self.observable by op.
        Find indices of new Paulis existing in self.observable (new_pauli_indices[new_pauli_in_observable]) and check if they exist in self.observable already (new_pauli_in_observable logical array).
        """
        new_paulis = obs[anticommuting_ind]
        new_paulis.compose_with(op)
        new_pauli_indices = obs.find_element_index(new_paulis)
        new_pauli_in_observable = obs.find_elements(new_paulis, new_pauli_indices)
        return new_paulis, new_pauli_indices[new_pauli_in_observable], new_pauli_in_observable

    @staticmethod
    def evolve_pauli_sum(pauli_list, operator_sequence, coeffs = None, **kwargs):
        """
        A wrapper for evaluating an expectation value of a sum of Paulis by first splitting the pauli_list into individual Paulis
        and constructing and running Simulation for each Pauli.
        """
        res = []
        for i, pauli in enumerate(pauli_list):
            c = np.array([1.0]) if coeffs is None else np.array(coeffs[i])
            res.append(Simulation.from_pauli_list(pauli, operator_sequence=operator_sequence, observable_coeffs=c, **kwargs).run_circuit())
        return sum([r for r in res])
