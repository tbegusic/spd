import numpy as np
from .utils import *
from .PauliRepresentation import *
from .SparsePauliDynamics import Simulation
from qiskit.quantum_info import SparsePauliOp

class DynamicsSimulation(Simulation):
    """
    Implements continuous-time sparse Pauli dynamics based on BCH or similar expansions.
    """
    def run_dynamics(self, nsteps, process = None, process_every = 1, method='euler-modified'):
        h = PauliRepresentation.from_sparse_pauli_op(self.operator_sequence)
        h.coeffs *= 2j

        if method=='euler-modified':
            apply_step = self.apply_step_euler_modified
            self.tan_coeffs = self.sin_coeffs / self.cos_coeffs
            h.coeffs = (1j) * self.tan_coeffs
        elif method=='euler':
            apply_step = self.apply_step_euler
        elif method=='rk2':
            apply_step = self.apply_step_rk2
        elif method=='rk4':
            apply_step = self.apply_step_rk4
        else:
            print('Method' + str(method) +' not implemented.')
            return

        r = []
        for step in range(nsteps+1):
            apply_step(h)
            if process is not None and (step % process_every == 0):
                r.append(process(self.observable))
        if process is not None:
            return r
    
    def apply_step_euler_modified(self, h):
        """
        Applies one step of dynamics with modified Euler method. 
        """
        anticommuting = self.observable.anticommutes_list(h)
        if (np.any(anticommuting)):

            mask_mult(self.observable.coeffs, self.cos_coeffs, anticommuting)
            new_paulis = self.observable.compose(h, anticommuting)

            self.observable.sum_with_threshold(new_paulis, self.threshold, self.nprocs==1)

    def apply_step_euler(self, h):
        """
        Applies one step of dynamics with Euler method. 
        """
        h_times_obs = self.apply_h(h) 
        self.observable.sum_with_threshold(h_times_obs, self.threshold, self.nprocs==1)

    def apply_step_rk2(self, h):
        """
        Applies one step of dynamics with Runge-Kutta 2nd-order method. 
        """
        k1 = self.observable.apply_h(h)
        y_plus_k1 = self.observable.copy()
        y_plus_k1.sum_with_threshold(k1, self.threshold, self.nprocs==1)
        k2 = y_plus_k1.apply_h(h)

        if k1 is not None and k2 is not None:
            k1_plus_k2_over_2 = PauliRepresentation(np.vstack((k1.bits, k2.bits)), np.hstack((k1.phase, k2.phase)), k1.nq, coeffs=np.hstack((k1.coeffs, k2.coeffs))/2)
        elif k1 is not None:
            pmult(k1.coeffs, 0.5)
            k1_plus_k2_over_2 = k1
        elif k2 is not None:
            pmult(k2.coeffs, 0.5)
            k1_plus_k2_over_2 = k2
        else:
            k1_plus_k2_over_2 = None
        self.observable.sum_with_threshold(k1_plus_k2_over_2, self.threshold, self.nprocs==1)

    def apply_step_rk4(self, h):
        """
        Applies one step of dynamics with Runge-Kutta 4th-order method. 
        """
        k1 = self.observable.apply_h(h)
        y1 = self.observable.copy()
        if k1 is not None:
            pmult(k1.coeffs, 0.5)
            y1.sum_with_threshold(k1, self.threshold, self.nprocs==1)

        k2 = y1.apply_h(h)
        y2 = self.observable.copy()
        if k2 is not None:
            pmult(k2.coeffs, 0.5)
            y2.sum_with_threshold(k2, self.threshold, self.nprocs==1)

        k3 = y2.apply_h(h)
        y3 = self.observable.copy()
        y3.sum_with_threshold(k3, self.threshold, self.nprocs==1)

        k4 = y3.apply_h(h)

        k_list_all = [k1, k2, k3, k4]
        k_list = []
        for i, k in enumerate(k_list_all):
            if k is not None:
                pmult(k.coeffs, [1/3, 2/3, 2/6, 1/6][i])
                k_list.append(k)
        if len(k_list):
            k_bits = np.vstack([k.bits for k in k_list])
            k_phase = np.hstack([k.phase for k in k_list])
            k_coeffs = np.hstack([k.coeffs for k in k_list])
            k_sum = PauliRepresentation(k_bits, k_phase, self.observable.nq, coeffs=k_coeffs)
            self.observable.sum_with_threshold(k_sum, self.threshold, self.nprocs==1)
