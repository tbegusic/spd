import numpy as np
from .utils import *
from .PauliRepresentation import *
from qiskit.quantum_info import SparsePauliOp

class Simulation:
    """
    Implements sparse Pauli dynamics .

    Attributes:
    -----------
    nq : number of qubits
    observable : Heisenberg-evolved observable represented as a sum of Paulis (PauliRepresentation class)
    operator_sequence : Sequence of Pauli rotation gates (OperatorSequence class)
    threshold : parameter for truncating the representation of the observable
    sin_coeffs, cos_coeffs : stores precomputed sine and cosine of rotation angles, which are used repeatedly
    eval_exp_val : A general function for evaluating the expectation value of a Pauli
    nprocs : number of processes for parallel runs (default value: 1)
    """
    def __init__(self, observable, operator_sequence, **kwargs):
        self.nq = observable.nq
        self.observable = observable
        if hasattr(self.observable, 'coeffs'):
            self.observable.coeffs = np.array(kwargs.get('observable_coeffs', self.observable.coeffs), dtype=np.complex128)
        else:
            self.observable.coeffs = np.array(kwargs.get('observable_coeffs', [1.0]), dtype=np.complex128)
        self.observable.coeffs = self.observable.coeffs[self.observable.order_pauli()]
        self.operator_sequence = operator_sequence
        self.threshold = kwargs.get('threshold', 0.01)
        self.sin_coeffs = np.sin(2*operator_sequence.coeffs)
        self.cos_coeffs = np.cos(2*operator_sequence.coeffs)
        self.eval_exp_val = kwargs.get('exp_val_fun', None)
        self.nprocs = kwargs.get('nprocs', 1)
        set_num_threads(self.nprocs)

        if self.eval_exp_val is None:
            self.eval_exp_val = evaluate_expectation_value_zero_state

    @classmethod
    def from_pauli_list(cls, observable, operator_sequence, **kwargs):
        if isinstance(observable, SparsePauliOp):
            plist = observable._pauli_list
            coeffs = observable.coeffs
            return cls(PauliRepresentation.from_pauli_list(plist), operator_sequence, observable_coeffs = coeffs, **kwargs)
        else:
            return cls(PauliRepresentation.from_pauli_list(observable), operator_sequence, **kwargs)
    
    def run(self):
        """
        Runs SPD simulation.
        Loops through all gates and applies each gate.
        """
        for j in range(self.operator_sequence.size):
            op = PauliRepresentation.from_pauli_list(self.operator_sequence._pauli_list[j])
            self.apply_gate(j, op)

    def run_circuit(self):
        """
        Runs a circuit simulation:
        First performs SPD propagation and then computes the expectation value and returns it.
        """
        self.run()
        nonzero_pauli_indices = np.where(self.observable.ztype())[0]
        return np.sum(self.observable.coeffs[nonzero_pauli_indices] * self.eval_exp_val(self.observable, nonzero_pauli_indices))

    def run_dynamics(self, nsteps, process = None, process_every = 1):
        r = []
        if process is not None:
            r.append(process(self.observable))
        for step in range(nsteps):
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
            new_paulis, new_pauli_indices, new_pauli_in_observable = self.prepare_new_paulis(self.observable, anticommuting, op)

            #Update coefficients for existing Paulis and precompute sin(theta) for those that will be added later.
            coeffs_sin = np.array(self.observable.coeffs[anticommuting])
            pmult(coeffs_sin, (1j) * self.sin_coeffs[j])
            update_coeffs(self.observable.coeffs, self.observable.coeffs[new_pauli_indices%self.observable.size()], self.cos_coeffs[j], (1j) * self.sin_coeffs[j], new_paulis.phase, self.observable.phase[new_pauli_indices%self.observable.size()], anticommuting, new_pauli_in_observable)

            #Project out Paulis and their coefficients that are below threshold.
            to_add_remove = np.empty(len(anticommuting), dtype=np.bool_)
            a_lt_b(self.observable.coeffs[anticommuting], self.threshold, to_add_remove)
            if np.any(to_add_remove):
                self.observable.delete_pauli(anticommuting[to_add_remove], self.nprocs==1)

            #Find which Paulis will be added to the observable.
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
