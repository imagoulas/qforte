"""
UCCNVQE classes
====================================
Classes for using an experiment to execute the variational quantum eigensolver
for a Trotterized (disentangeld) UCCN ansatz with fixed operators.
"""

import qforte
from qforte.abc.uccvqeabc import UCCVQE

from qforte.experiment import *
from qforte.maths.optimizer import diis
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize

import numpy as np
from scipy.optimize import minimize

class UCCNVQE(UCCVQE):
    """A class that encompasses the three components of using the variational
    quantum eigensolver to optimize a parameterized disentangled UCCN-like
    wave function. (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy and
    gradients (3) optemizes the the wave funciton by minimizing the energy

    Attributes
    ----------
    _results : list
        The optimizer result objects from each iteration of UCCN-VQE.

    _energies : list
        The optimized energies from each iteration of UCCN-VQE.

    _grad_norms : list
        The gradient norms from each iteration of UCCN-VQE.

    """
    def run(self,
            opt_thresh=1.0e-5,
            opt_ftol=1.0e-5,
            opt_maxiter=200,
            pool_type='SD',
            optimizer='BFGS',
            shift=0.0,
            use_analytic_grad = True,
            noise_factor = 0.0,
            stupid = False):

        self._stupid = stupid
        self._opt_thresh = opt_thresh
        self._opt_ftol = opt_ftol
        self._opt_maxiter = opt_maxiter
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        self._shift = shift
        self._pool_type = pool_type
        self._noise_factor = noise_factor

        self._tops = []
        self._tamps = []
        self._conmutator_pool = []
        self._converged = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        self._res_vec_evals = 0
        self._res_m_evals = 0
        self._k_counter = 0

        self._curr_grad_norm = 0.0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### UCCN-VQE #########

        self.fill_pool()

        if self._verbose:
            print(self._pool_obj.str())

        self.initialize_ansatz()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('\nInitial tamplitudes for tops: \n', self._tamps)

        self.solve()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('\nFinal tamplitudes for tops: \n', self._tamps)

        ######### UCCSD-VQE #########
        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if(np.abs(tmu) > 1.0e-12):
                self._n_nonzero_params += 1

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

        self.print_summary_banner()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for UCCN-VQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('          Unitary Coupled Cluster VQE   ')
        print('-----------------------------------------------------')

        print('\n\n               ==> UCCN-VQE options <==')
        print('---------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        print('Use compact excitation circuits:         ',  self._compact_excitations)
        print('Use qubit excitations:                   ',  self._qubit_excitations)
        if(self._fast):
            print('Measurement variance thresh:             ',  'NA')
        else:
            print('Measurement variance thresh:             ',  0.01)


        # VQE options.
        opt_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        print('Optimization algorithm:                  ',  self._optimizer)
        print('Optimization maxiter:                    ',  self._opt_maxiter)
        print('Optimizer grad-norm threshold (theta):   ',  opt_thrsh_str)

        # UCCVQE options.
        print('Use analytic gradient:                   ',  str(self._use_analytic_grad))
        print('Operator pool type:                      ',  str(self._pool_type))


    def print_summary_banner(self):

        print('\n\n                ==> UCCN-VQE summary <==')
        print('-----------------------------------------------------------')
        print('Final UCCN-VQE Energy:                      ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._pool_obj))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Total number of Hamiltonian measurements:    ', self.get_num_ham_measurements())
        print('Total number of commutator measurements:     ', self.get_num_commut_measurements())
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)

        print('Number of grad vector evaluations:           ', self._res_vec_evals)
        print('Number of individual grad evaluations:       ', self._res_m_evals)

    def solve(self):
            return self.scipy_solve()

    def scipy_solve(self):
        if not self._stupid:
            # Construct arguments to hand to the minimizer.
            opts = {}

            # Options common to all minimization algorithms
            opts['disp'] = True
            opts['maxiter'] = self._opt_maxiter

            # Optimizer-specific options
            if self._optimizer.lower() in ['bfgs', 'cg', 'l-bfgs-b', 'tnc', 'trust-constr']:
                opts['gtol'] = self._opt_thresh
            if self._optimizer.lower() == 'nelder-mead':
                opts['fatol'] = self._opt_ftol
            if self._optimizer.lower() in ['powell', 'l-bfgs-b', 'tnc', 'slsqp']:
                opts['ftol'] = self._opt_ftol
            if self._optimizer.lower() in ['l-bfgs-b', 'tnc']:
                opts['maxfun']  = self._opt_maxiter

            x0 = copy.deepcopy(self._tamps)
            init_gues_energy = self.energy_feval(x0)
            self._prev_energy = init_gues_energy

            if self._use_analytic_grad:
                print('  \n--> Begin opt with analytic gradient:')
                print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
                res =  minimize(self.energy_feval, x0,
                                        method=self._optimizer,
                                        jac=self.gradient_ary_feval,
                                        tol=self._opt_thresh,
                                        options=opts,
                                        callback=self.report_iteration)

                # account for paulit term measurement for gradient evaluations
                # for m in range(len(self._tamps)):
                #     self._n_pauli_trm_measures += self._Nm[m] * self._Nl * res.njev

                if hasattr(res, 'njev'):
                    for tmu in res.x:
                        if(np.abs(tmu) > 1.0e-12):
                            self._n_pauli_trm_measures += int(2 * self._Nl * res.njev)

                self._n_pauli_trm_measures += int(self._Nl * res.nfev)


            else:
                print('  \n--> Begin opt with grad estimated using first-differences:')
                print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
                res =  minimize(self.energy_feval, x0,
                                        method=self._optimizer,
                                        tol=self._opt_thresh,
                                        options=opts,
                                        callback=self.report_iteration)

                # account for pauli term measurement for energy evaluations
                self._n_pauli_trm_measures += self._Nl * res.nfev

            if(res.success):
                print('  => Minimization successful!')
            else:
                print('  => WARNING: minimization result may not be tightly converged.')
            print(f'  => Minimum Energy: {res.fun:+12.10f}')
            self._Egs = res.fun
            if(self._optimizer.lower() == 'powell'):
                self._Egs = res.fun[()]
            self._final_result = res
            self._tamps = list(res.x)

            self._n_classical_params = len(self._tamps)
            self._n_cnot = self.build_Uvqc().get_num_cnots()

        else:
            # Construct arguments to hand to the minimizer.
            opts = {}

            # Options common to all minimization algorithms
            opts['disp'] = False
            opts['maxiter'] = 1

            # Optimizer-specific options
            if self._optimizer.lower() in ['bfgs', 'cg', 'l-bfgs-b', 'tnc', 'trust-constr']:
                opts['gtol'] = self._opt_thresh
            if self._optimizer.lower() == 'nelder-mead':
                opts['fatol'] = self._opt_ftol
            if self._optimizer.lower() in ['powell', 'l-bfgs-b', 'tnc', 'slsqp']:
                opts['ftol'] = self._opt_ftol
            if self._optimizer.lower() in ['l-bfgs-b', 'tnc']:
                opts['maxfun']  = self._opt_maxiter

            self._t_diis = [copy.deepcopy(self._tamps)]
            self._e_diis = []

            print('  \n--> Begin opt with analytic gradient:')
            print(f" Initial guess energy:              {self.energy_feval(self._tamps):+12.10f}")

            for k in range(1, self._opt_maxiter+1):
                x0 = copy.deepcopy(self._tamps)
                self._prev_energy = self.energy_feval(x0)

                res =  minimize(self.energy_feval, x0,
                                        method=self._optimizer,
                                        jac=self.gradient_ary_feval,
                                        tol=self._opt_thresh,
                                        options=opts)

                if(k == 1):

                    print('\n    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
                    print('--------------------------------------------------------------------------------------------------')
                    if (self._print_summary_file):
                        f = open("summary.dat", "w+", buffering=1)
                        f.write('\n#    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
                        f.write('\n#--------------------------------------------------------------------------------------------------')
                        f.close()

                # else:
                dE = self._curr_energy - self._prev_energy
                print(f'     {k:7}        {self._curr_energy:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.10f}')

                if (self._print_summary_file):
                    f = open("summary.dat", "a", buffering=1)
                    f.write(f'\n       {k:7}        {self._curr_energy:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.12f}')
                    f.close()

                self._prev_energy = self._curr_energy

                self._tamps = list(res.x)

                if self._optimizer.lower() in ['nelder-mead', 'powell', 'cobyla']:
                    if abs(dE) <= self._opt_thresh:
                        break
                elif self._curr_grad_norm <= self._opt_thresh:
                    break

                self._t_diis.append(copy.deepcopy(res.x))
                self._e_diis.append(np.subtract(self._t_diis[-1], self._t_diis[-2]))

                if(k >= 1 and self._diis_max_dim >= 2):
                    self._tamps = diis(self._diis_max_dim, self._t_diis, self._e_diis)

                # account for paulit term measurement for gradient evaluations
                # for m in range(len(self._tamps)):
                #     self._n_pauli_trm_measures += self._Nm[m] * self._Nl * res.njev

            if hasattr(res, 'njev'):
                for tmu in res.x:
                    if(np.abs(tmu) > 1.0e-12):
                        self._n_pauli_trm_measures += int(2 * self._Nl * res.njev)

            self._Egs = res.fun
            if(self._optimizer.lower() == 'powell'):
                self._Egs = res.fun[()]
            self._final_result = res
            self._tamps = list(res.x)

            self._n_classical_params = len(self._tamps)
            self._n_cnot = self.build_Uvqc().get_num_cnots()

            self._n_pauli_trm_measures += int(self._Nl * res.nfev)



    def initialize_ansatz(self):
        """Adds all operators in the pool to the list of operators in the circuit,
        with amplitude 0.
        """
        self._tops = list(range(len(self._pool_obj)))
        self._tamps = [0.0] * len(self._pool_obj)

    # TODO: change to get_num_pt_evals
    def get_num_ham_measurements(self):
        """Returns the total number of times the energy was evaluated via
        measurement of the Hamiltonian.
        """
        try:
            self._n_ham_measurements = self._final_result.nfev
            return self._n_ham_measurements
        except AttributeError:
            # TODO: Determine the number of Hamiltonian measurements
            return "Not Yet Implemented"

    # TODO: depricate this function
    def get_num_commut_measurements(self):
        # if self._use_analytic_grad:
        #     self._n_commut_measurements = self._final_result.njev * (len(self._pool_obj))
        #     return self._n_commut_measurements
        # else:
        #     return 0
        return 0
