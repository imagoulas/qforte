"""
SPQE classes
====================================
Classes for implementing the selected variant of the projetive quantum eigensolver
"""

import qforte as qf

from qforte.abc.uccpqeabc import UCCPQE
from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.utils.point_groups import sq_op_find_symmetry
from qforte.maths import optimizer

import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from copy import deepcopy

class SPQE(UCCPQE):
    """This class implements the selected projective quantum eigensolver (SPQE) for
    disentagled UCC like ansatz.
    In SPQE, a batch of imporant particle-hole operators
    :math:`\{ e^{t_\mu (\hat{\\tau}_\mu - \hat{\\tau}_\mu^\dagger )} \}` are
    added at each macro-iteration :math:`n` to the SPQE unitary :math:`\hat{U}(\mathbf{t})`,
    wile all current parameters are optemized using the quasi-Newton PQE update
    with micro-iterations :math:`k`.

    In our selection approach we consider a (normalized) quantum state of the form

    .. math::
        | \\tilde{r} \\rangle  = \\tilde{r}_0 | \Phi_0 \\rangle + \sum_\mu \\tilde{r}_\mu  | \Phi_\mu \\rangle

    where the quantities :math:`\\tilde{r}_\mu` are approximately proportional to
    the residuals :math:`r_\mu`.
    The state :math:`| \\tilde{r} \\rangle` can be approximately reproduced via

    .. math::
        | \\tilde{r} \\rangle \\approx \hat{U}^\dagger e^{i \Delta t \hat{H}} \hat{U} | \Phi_0 \\rangle

    .. math::
        \\approx (1 + i\Delta t \hat{U}^\dagger \hat{H} \hat{U})  | \Phi_0 \\rangle + \mathcal{O}(\Delta t^2).

    We note that in this implementation we use a Trotter approximation for the time
    evolution unitary.
    Measuring :math:`\\langle \hat{Z} \\rangle` for each qubit yields a bitstring
    that has corresponding determinat and operator
    :math:`(\hat{\\tau}_\mu - \hat{\\tau}_\mu^\dagger )`
    with probablility proportional to :math:`|\\tilde{r}_\mu|^2`.
    The operators corresponding to the largest :math:`|\\tilde{r}_\mu|^2` values
    are then added to :math:`\hat{U}(\mathbf{t})` at each macro-iteration.
    """
    def run(self,
            spqe_thresh=1.0e-2,
            spqe_maxiter=20,
            dt=0.001,
            M_omega = 'inf',
            opt_thresh = 1.0e-5,
            opt_maxiter = 30,
            shift = 0.0,
            max_excit_rank = None,
            repeated_SD_pool=False,
            optimizer='Jacobi',
            mmcc = False,
            excit_state_idx = 0,
            Sz = 0,
            n_electrons = None,
            use_cumulative_thresh=True):

        if(self._state_prep_type != 'occupation_list'):
            raise ValueError("SPQE implementation can only handle occupation_list Hartree-Fock reference.")

        self._spqe_thresh = spqe_thresh
        self._spqe_maxiter = spqe_maxiter
        self._dt = dt
        if(M_omega != 'inf'):
            self._M_omega = int(M_omega)
        else:
            self._M_omega = M_omega

        self._use_cumulative_thresh = use_cumulative_thresh
        self._optimizer = optimizer
        self._opt_thresh = opt_thresh
        self._opt_maxiter = opt_maxiter
        self._shift = shift
        self._repeated_SD_pool = repeated_SD_pool
        self._mmcc = mmcc
        if self._mmcc:
            self._mpdenom = []
            self._E_mmcc_mp = []
            self._E_mmcc_en = []
            self._mmcc_aux_excitation_indices = []
            self._mmcc_aux_pool = qf.SQOpPool()
        self._excit_state_idx = excit_state_idx
        if n_electrons == None:
            self._n_electrons = sum(self._ref)
        elif not isinstance(n_electrons, int) or n_electrons < 1 or n_electrons > self._nqb:
            raise TypeError("The number of electrons must be a positive integer less or equal to the number of spin-orbitals")
        else:
            self._n_electrons = n_electrons
        if not isinstance(Sz, (int, float)) or Sz % 0.5 != 0.0 or abs(2 * Sz) > self._nqb:
            raise TypeError("For this particular system, Sz can take values between " +
                    str(-min(self._n_electrons, self._nqb - self._n_electrons) * 0.5) + " and " +
                    str(min(self._n_electrons, self._nqb - self._n_electrons) * 0.5) + " with a step of 1!")
        else:
            self._Sz = Sz

        self._total_spin_squared = []

        self._nbody_counts = []
        self._n_classical_params_lst = []

        self._results = []
        self._energies = []
        self._grad_norms = []
        self._tops = []
        self._tamps = []
        self._converged = False
        self._res_vec_evals = 0
        self._res_m_evals = 0

        self._curr_energy = 0.0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_cnot_lst = []
        self._n_pauli_trm_measures = 0
        self._n_pauli_trm_measures_lst = []

        self._eiH, self._eiH_phase = trotterize(self._qb_ham, factor= self._dt*(0.0 + 1.0j), trotter_number=self._trotter_number)

        self.build_orb_energies()

        for occupation in self._ref:
            if occupation:
                self._nbody_counts.append(0)

        if self._excit_state_idx != 0:
            if not isinstance(self._excit_state_idx, int) or self._excit_state_idx < 0:
                raise TypeError("The excited state index must be a non-negative integer!")

            diag_h = []

            ref = sum([b << i for i, b in enumerate(self._ref)])
            mask_alpha = 0x5555555555555555
            mask_beta = mask_alpha << 1
            for I in range(1 << self._nqb):
                alphas = [int(j) for j in bin(I & mask_alpha)[2:]]
                betas = [int(j) for j in bin(I & mask_beta)[2:]]
                if sum(alphas) + sum(betas) == self._n_electrons and (sum(alphas) - sum(betas))*0.5 == self._Sz:
                    if sq_op_find_symmetry(self._sys.orb_irreps_to_int,
                                           [len(alphas) - i - 1 for i, x in enumerate(alphas) if x],
                                           [len(betas) -i - 1 for i, x in enumerate(betas) if x]) == self._irrep:
                        excit = bin(ref ^ I).replace("0b", "")
                        if excit != "0":
                            occ_idx = [int(i) for i,j in enumerate(reversed(excit)) if int(j) == 1 and self._ref[i] == 1]
                            unocc_idx = [int(i) for i,j in enumerate(reversed(excit)) if int(j) == 1 and self._ref[i] == 0]
                            qc = qf.Computer(self._nqb)
                            qc.apply_circuit(self._Uprep)
                            for i in occ_idx + unocc_idx:
                                qc.apply_gate(qf.gate('X', i))
                            diag_h.append([np.real(qc.direct_op_exp_val(self._qb_ham)), occ_idx, unocc_idx])

            diag_h = sorted(diag_h)

            for i in diag_h[self._excit_state_idx-1][1]:
                self._ref[i] = 0
            for i in diag_h[self._excit_state_idx-1][2]:
                self._ref[i] = 1

            if self._Sz == 0:
                ref_alpha = []
                ref_beta = []
                for i in range(0, len(self._ref), 2):
                    ref_alpha.append(self._ref[i])
                for i in range(1, len(self._ref), 2):
                    ref_beta.append(self._ref[i])

                if ref_alpha != ref_beta:
                    spin_complement = 0
                    for i in range(len(ref_alpha)):
                        spin_complement += ref_beta[i] << (2*i)
                        spin_complement += ref_alpha[i] << (2*i + 1)

            self._Uprep = qf.build_Uprep(self._ref, self._state_prep_type)

        # create a pool of particle number, Sz, and spatial symmetry adapted second quantized operators
        # of maximum excitation rank max_excit_rank
        ref = sum([b << i for i, b in enumerate(self._ref)])
        mask_alpha = 0x5555555555555555
        mask_beta = mask_alpha << 1
        nalpha = sum(self._ref[0::2])
        nbeta = sum(self._ref[1::2])
        if max_excit_rank is None or repeated_SD_pool:
            max_excit_rank = nalpha + nbeta
        elif not isinstance(max_excit_rank, int) or max_excit_rank <= 0:
            raise TypeError("The maximum excitation rank max_excit_rank must be a positive integer!")
        elif max_excit_rank > nalpha + nbeta:
            max_excit_rank = nalpha + nbeta
            print("\nWARNING: The entered maximum excitation rank exceeds the number of particles.\n"
                    "         Procceding with max_excit_rank = {0}.\n".format(max_excit_rank))
        self._pool_type = max_excit_rank
        idx = 0
        # determinant id : excitation operator index
        self._excitation_dictionary = {}
        # list that holds the ids of determinants corresponding to operators in the pool
        self._excitation_indices = []
        self._pool_obj = qf.SQOpPool()
        if repeated_SD_pool:
            # the auxiliary operator pool contains the operators that generate the determinants
            # that span the N-electron Hilbert space; it is needed to obtain the correct signs
            # of the residuals
            self._aux_pool_obj = qf.SQOpPool()
        for I in range(1 << self._nqb):
            alphas = [int(j) for j in bin(I & mask_alpha)[2:]]
            betas = [int(j) for j in bin(I & mask_beta)[2:]]
            if sum(alphas) == nalpha and sum(betas) == nbeta:
                if sq_op_find_symmetry(self._sys.orb_irreps_to_int,
                                       [len(alphas) - i - 1 for i, x in enumerate(alphas) if x],
                                       [len(betas) -i - 1 for i, x in enumerate(betas) if x]) == self._irrep:
                    excit = bin(ref ^ I).replace("0b", "")
                    if excit != "0":
                        if int(excit.count('1')/2) <= self._pool_type:
                            occ_idx = [int(i) for i,j in enumerate(reversed(excit)) if int(j) == 1 and self._ref[i] == 1]
                            unocc_idx = [int(i) for i,j in enumerate(reversed(excit)) if int(j) == 1 and self._ref[i] == 0]
                            if self._mmcc:
                                self._mpdenom.append(sum(self._orb_e[x] for x in occ_idx) - sum(self._orb_e[x] for x in unocc_idx))
                            if repeated_SD_pool:
                                sq_op = qf.SQOperator()
                                sq_op.add(+1.0, unocc_idx, occ_idx)
                                sq_op.add(-1.0, occ_idx[::-1], unocc_idx[::-1])
                                sq_op.simplify()
                                self._aux_pool_obj.add_term(0.0, sq_op)
                                if len(occ_idx) > 2:
                                    # find the double excitation that obeys the symmetry criteria and whose quantum circuit
                                    # requires the smallest number of CNOT gates
                                    distance = self._nqb - 1
                                    for occ_comb in combinations(occ_idx, 2):
                                        for unocc_comb in combinations(unocc_idx, 2):
                                            if len([i for i in occ_comb if i%2==0]) == len([i for i in unocc_comb if i%2==0]):
                                                if qf.sq_op_find_symmetry(self._sys.orb_irreps_to_int, occ_comb, unocc_comb) == 0:
                                                    if max(unocc_comb) - min(occ_comb) <= distance:
                                                        distance = max(unocc_comb) - min(occ_comb)
                                                        occ_idx = list(occ_comb)
                                                        unocc_idx = list(unocc_comb)
                            sq_op = qf.SQOperator()
                            sq_op.add(+1.0, unocc_idx, occ_idx)
                            sq_op.add(-1.0, occ_idx[::-1], unocc_idx[::-1])
                            sq_op.simplify()
                            self._pool_obj.add_term(0.0, sq_op)
                            self._excitation_dictionary[I] = idx
                            self._excitation_indices.append(I)
                            if self._excit_state_idx != 0 and self._Sz == 0 and ref_alpha != ref_beta:
                                if I == spin_complement:
                                    self._tops.append(idx)
                                    self._tamps.append(np.pi/4)
                            if self._mmcc:
                                self._mmcc_aux_pool.add_term(0.0, sq_op)
                                self._mmcc_aux_excitation_indices.append(I)
                            idx += 1

                        if int(excit.count('1')/2) > self._pool_type and self._mmcc:
                            occ_idx = [int(i) for i,j in enumerate(reversed(excit)) if int(j) == 1 and self._ref[i] == 1]
                            unocc_idx = [int(i) for i,j in enumerate(reversed(excit)) if int(j) == 1 and self._ref[i] == 0]
                            sq_op = qf.SQOperator()
                            sq_op.add(+1.0, unocc_idx, occ_idx)
                            sq_op.add(-1.0, occ_idx[::-1], unocc_idx[::-1])
                            sq_op.simplify()
                            self._mmcc_aux_pool.add_term(0.0, sq_op)
                            self._mpdenom.append(sum(self._orb_e[x] for x in occ_idx) - sum(self._orb_e[x] for x in unocc_idx))
                            self._mmcc_aux_excitation_indices.append(I)


        if self._mmcc:
            self._epstein_nesbet = []
            for i in self._mmcc_aux_pool.terms():
                sq_op = i[1]
                qc = qf.Computer(self._nqb)
                qc.apply_circuit(self._Uprep)
                qc.apply_operator(sq_op.jw_transform(self._qubit_excitations))
                self._epstein_nesbet.append(qc.direct_op_exp_val(self._qb_ham))

        # excitation operator index : determinant id
        self._reversed_excitation_dictionary = {value: key for key, value in self._excitation_dictionary.items()}

        self.print_options_banner()

        self._spqe_iter = 1

        if(self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write(f"#{'Iter':>8}{'E':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')

        while not self._converged:

            self.update_ansatz()

            if self._converged:
                break

            if(self._verbose):
                print('\ntoperators included from pool: \n', self._tops)
                print('\ntamplitudes for tops: \n', self._tamps)

            self.solve()

            # Compute epxectation value of S^2
            U = self.ansatz_circuit()
            comp = qf.Computer(self._nqb)
            comp.apply_circuit(self._Uprep)
            comp.apply_circuit(U)
            self._total_spin_squared.append(comp.direct_op_exp_val(qf.total_spin_squared(self._nqb)).real)

            if(self._verbose):
                print('\ntamplitudes for tops post solve: \n', np.real(self._tamps))

            if(self._print_summary_file):
                f.write(f'  {self.spqe_iter:7}    {self._energies[-1]:+15.9f}    {len(self._tamps):8}        {self._n_cnot_lst[-1]:10}        {sum(self._n_pauli_trm_measures_lst[:k+1]):12}\n')
            self._spqe_iter += 1


        if(self._print_summary_file):
            f.close()

        self._Egs = self._energies[-1]

        print("\n\n")
        print("---> Final n-body excitation counts in SPQE ansatz <---")
        print("\n")
        print(f"{'Excitaion order':>20}{'Number of operators':>30}")
        print('---------------------------------------------------------')
        for l, nl in enumerate(self._nbody_counts):
            print(f"{l+1:12}              {nl:14}")

        print('\n\n')
        if not self._mmcc:
            print(f"{'Iter':>8}{'E':>14}{'<S^2>':>11}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}")
            print('-------------------------------------------------------------------------------')

            for k, Ek in enumerate(self._energies):
                print(f' {k+1:7}    {Ek:+15.9f}    {self._total_spin_squared[k]:+7.4f}    {self._n_classical_params_lst[k]:8}        {self._n_cnot_lst[k]:10}        {sum(self._n_pauli_trm_measures_lst[:k+1]):12}')
        else:
            print(f"{'Iter':>8}{'E':>14}{'E_MMCC(MP)':>24}{'E_MMCC(EN)':>19}{'<S^2>':>11}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
            print('-----------------------------------------------------------------------------------------------------------')

            for k, Ek in enumerate(self._energies):
                print(f' {k+1:7}    {Ek:+15.9f}    {self._E_mmcc_mp[k]:15.9f}    {self._E_mmcc_en[k]:15.9f}    {self._total_spin_squared[k]:+7.4f}    {self._n_classical_params_lst[k]:8}        {self._n_cnot_lst[k]:10}        {sum(self._n_pauli_trm_measures_lst[:k+1]):12}')

        self._n_classical_params = len(self._tamps)
        self._n_cnot = self._n_cnot_lst[-1]
        self._n_pauli_trm_measures = sum(self._n_pauli_trm_measures_lst)

        self.print_summary_banner()
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for SPQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_PQE_attributes()
        self.verify_required_UCCPQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('       Selected Projective Quantum Eigensolver   ')
        print('-----------------------------------------------------')

        print('\n\n               ==> SPQE options <==')
        print('---------------------------------------------------------')
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of electrons:                     ',  self._n_electrons)
        print('Sz:                                      ',  self._Sz)
        print('State irreducible representation:        ',  self._sys._point_group[1][self._irrep])
        if self._excit_state_idx == 0:
            print('Excited state index:                      0 (ground state)')
        else:
            print('Excited state index:                     ', self._excit_state_idx, '(excited state)')
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        print('Use compact excitation circuits:         ',  self._compact_excitations)
        print('Use qubit excitations:                   ',  self._qubit_excitations)
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        opt_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        spqe_thrsh_str = '{:.2e}'.format(self._spqe_thresh)
        print('Optimizer:                               ', self._optimizer)
        if self._diis_max_dim >=2:
            print('DIIS dimension:                          ', self._diis_max_dim)
        else:
            print('DIIS dimension:                          Disabled')
        print('Compute MMCC energy correction:          ', self._mmcc)
        print('Number of micro-iterations:              ',  self._opt_maxiter)
        print('Micro-iteration residual-norm threshold (omega_r):  ',  opt_thrsh_str)
        print('Maximum excitation rank in operator pool:',  self._pool_type)
        print('Use a repeated SD operator pool:         ', self._repeated_SD_pool)
        print('Number of operators in pool:             ', len(self._pool_obj))
        print('SPQE residual-norm threshold (Omega):    ',  spqe_thrsh_str)
        print('SPQE maxiter:                            ',  self._spqe_maxiter)


    def print_summary_banner(self):
        print('\n\n                ==> SPQE summary <==')
        print('-----------------------------------------------------------')
        print('Final SPQE Energy:                           ', round(self._Egs, 10))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of individual residual evaluations:   ', self._res_m_evals)

    def solve(self):
        if self._optimizer.lower() == 'jacobi':
            self.jacobi_solver()
        elif self._optimizer.lower() in ['nelder-mead', 'powell', 'bfgs', 'l-bfgs-b', 'cg', 'slsqp']:
            self.scipy_solve()
        else:
           raise NotImplementedError('Currently only Jacobi, Nelder-Mead, Powell, and BFGS solvers are implemented')


    def scipy_solve(self):

        # Construct arguments to hand to the minimizer.
        opts = {}

        # Options common to all minimization algorithms
        opts['disp'] = True
        opts['maxiter'] = self._opt_maxiter

        # Optimizer-specific options
        if self._optimizer.lower() in ['bfgs', 'cg', 'l-bfgs-b']:
            opts['gtol'] = self._opt_thresh
        if self._optimizer.lower() == 'nelder-mead':
            opts['fatol'] = self._opt_thresh
            opts['adaptive'] = True
        if self._optimizer.lower() in ['powell', 'l-bfgs-b', 'slsqp']:
            opts['ftol'] = self._opt_thresh

        x0 = copy.deepcopy(self._tamps)
        self._prev_energy = self.energy_feval(x0)
        self._k_counter = 0

        #if self._optimizer.lower() == 'nelder-mead':
        #    res = minimize(self.get_sum_residual_square, x0,
        #            method='BFGS',
        #            options={'maxiter' : 1})

        #    x0 = list(res.x)


        res = minimize(self.get_sum_residual_square, x0,
                method=self._optimizer,
                options=opts,
                callback=self.report_iteration)

        if(res.success):
            print('  => Minimization successful!')
        else:
            print('  => WARNING: minimization result may not be tightly converged.')

        self._tamps = list(res.x)
        self._Egs = self.energy_feval(self._tamps)
        self._energies.append(self._Egs)
        self._n_pauli_measures_k += self._Nl*self._k_counter * (2*len(self._tamps) + 1)
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
        self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())

    def get_residual_vector(self, trial_amps):
        U = self.ansatz_circuit(trial_amps)

        qc_res = qforte.Computer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_operator(self._qb_ham)
        qc_res.apply_circuit(U.adjoint())

        coeffs = qc_res.get_coeff_vec()
        residuals = []

        for m in self._tops:

            if self._optimizer.lower() == 'jacobi':
                if self._repeated_SD_pool:
                    sq_op = self._aux_pool_obj[m][1]
                else:
                    sq_op = self._pool_obj[m][1]
                # occ => i,j,k,...
                # vir => a,b,c,...
                # sq_op is 1.0(a^ b^ i j) - 1.0(j^ i^ b a)

                qc_temp = qforte.Computer(self._nqb)
                qc_temp.apply_circuit(self._Uprep)
                qc_temp.apply_operator(sq_op.jw_transform(self._qubit_excitations))
                sign_adjust = qc_temp.get_coeff_vec()[self._reversed_excitation_dictionary[m]]

                res_m = coeffs[self._reversed_excitation_dictionary[m]] * sign_adjust
                if(np.imag(res_m) > 0.0):
                    raise ValueError("residual has imaginary component, someting went wrong!!")
                residuals.append(res_m.real)
            else:
                residuals.append(coeffs[self._reversed_excitation_dictionary[m]].real)


        self._res_vec_norm = np.linalg.norm(residuals)
        self._res_vec_evals += 1
        self._res_m_evals += len(trial_amps)

        return residuals

    def get_sum_residual_square(self, tamps):
        residual_vector = self.get_residual_vector(tamps)
        sum_residual_vector_square = np.sum(np.square(residual_vector))
        return sum_residual_vector_square


    def update_ansatz(self):
        self._n_pauli_measures_k = 0
        # TODO: Check if this deepcopy is needed. The one argument of energy_feval should be const.
        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)

        # do U^dag e^iH U |Phi_o> = |Phi_res>
        U = self.ansatz_circuit()

        qc_res = qf.Computer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_circuit(self._eiH)
        qc_res.apply_circuit(U.adjoint())


        res_coeffs = [i / self._dt for i in qc_res.get_coeff_vec()]

        # build different res_sq list using M_omega
        if(self._M_omega != 'inf'):
            res_sq_tmp = [ np.real(np.conj(res_coeffs[I]) * res_coeffs[I]) for I in range(len(res_coeffs))]

            # Nmu_lst => [ det1, det2, det3, ... det_M_omega]
            det_lst = np.random.choice(len(res_coeffs), self._M_omega, p=res_sq_tmp * self._dt * self._dt)

            print(f'|Co|dt^2 :       {np.amax(res_sq_tmp):12.14f}')
            print(f'mu_o :           {np.where(res_sq_tmp == np.amax(res_sq_tmp))[0][0]}')

            No_idx = np.where(res_sq_tmp == np.amax(res_sq_tmp))[0][0]
            print(f'\nNo_idx   {No_idx:4}')

            No = np.count_nonzero(det_lst == No_idx)
            print(f'\nNo       {No:10}')

            res_sq = []
            Nmu_lst = []
            for mu in range(len(res_coeffs)):
                Nmu = np.count_nonzero(det_lst == mu)
                if(Nmu > 0):
                    print(f'mu:    {mu:8}      Nmu      {Nmu:10}  r_mu: { Nmu / (self._M_omega):12.14f} ')
                    Nmu_lst.append((Nmu, mu))
                res_sq.append( ( Nmu / (self._M_omega), mu) )

            ## 1. sort
            Nmu_lst.sort()
            res_sq.sort()

            ## 2. set norm
            self._curr_res_sq_norm = sum(rmu_sq[0] for rmu_sq in res_sq[:-1]) / (self._dt * self._dt)

            ## 3. print stuff
            print('  \n--> Begin selection opt with residual magnitudes:')
            print('  Initial guess energy:          ', round(init_gues_energy,10))
            print(f'  Norm of approximate res vec:  {np.sqrt(self._curr_res_sq_norm):14.12f}')

            ## 4. check conv status (need up update function with if(M_omega != 'inf'))
            if(len(Nmu_lst)==1):
                print('  SPQE converged with M_omega thresh!')
                self._converged = True
                self._final_energy = self._energies[-1]
                #self._final_result = self._results[-1]
            else:
                self._converged = False

            ## 5. add new toperator
            if not self._converged:
                if self._verbose:
                    print('\n')
                    print('     op index (Imu)     Number of times measured')
                    print('  -----------------------------------------------')

                for Nmu_tup in Nmu_lst[:-1]:
                    if(self._verbose):
                        print(f"  {Nmu_tup[1]:10}                  {np.real(Nmu_tup[0]):14}")
                    if(Nmu_tup[1] not in self._tops):
                        self._tops.insert(0,self._excitation_dictionary[Nmu_tup[1]])
                        self._tamps.insert(0,0.0)
                        self._nbody_counts[len(self._pool_obj[self._excitation_dictionary[Nmu_tup[1]]][1].terms()[0][1]) - 1] += 1

                self._n_classical_params_lst.append(len(self._tops))

        else: # when M_omega == 'inf', proceed with standard SPQE
            res_sq = [( np.real(np.conj(res_coeffs[I]) * res_coeffs[I]), I) for I in set(self._excitation_indices) - {self._reversed_excitation_dictionary[i] for i in self._tops}]
            res_sq.sort()
            self._curr_res_sq_norm = sum(rmu_sq[0] for rmu_sq in res_sq)

            self.conv_status()

            if not self._converged:
                print('\n\n -----> SPQE iteration ', self._spqe_iter, ' <-----\n')
                print('  \n--> Begin selection opt with residual magnitudes |r_mu|:')
                print('  Initial guess energy: ', round(init_gues_energy,10))
                print(f'  Norm of res vec:      {np.sqrt(self._curr_res_sq_norm):14.12f}')

                if self._verbose:
                    print('\n')
                    print('     op index (Imu)           Residual Factor')
                    print('  -----------------------------------------------')
                res_sq_sum = 0.0

                if(self._use_cumulative_thresh):
                    # Make a running list of operators. When the sum of res_sq exceeds the target, every operator
                    # from here out is getting added to the ansatz..
                    temp_ops = []
                    for rmu_sq, op_idx in res_sq:
                        res_sq_sum += rmu_sq
                        if res_sq_sum > (self._spqe_thresh * self._spqe_thresh):
                            if self._verbose:
                                print(f"  {self._excitation_dictionary[op_idx]:10}                  {np.real(rmu_sq):14.12f}"
                                      f"   {self._pool_obj[self._excitation_dictionary[op_idx]][1].str()}" )
                            if self._excitation_dictionary[op_idx] not in self._tops:
                                temp_ops.append(self._excitation_dictionary[op_idx])
                                self._nbody_counts[len(self._pool_obj[self._excitation_dictionary[op_idx]][1].terms()[0][1]) - 1] += 1

                    for temp_op in temp_ops[::-1]:
                        self._tops.insert(0, temp_op)
                        self._tamps.insert(0, 0.0)

                else:
                    # Add the single operator with greatest rmu_sq not yet in the ansatz
                    res_sq.reverse()
                    for rmu_sq, op_idx in res_sq:
                        print(f"  {self._excitation_dictionary[op_idx]:10}                  {np.real(rmu_sq):14.12f}")
                        if self._excitation_dictionary[op_idx] not in self._tops:
                            print('Adding this operator to ansatz')
                            self._tops.insert(0, self._excitation_dictionary[op_idx])
                            self._tamps.insert(0, 0.0)
                            self._nbody_counts[len(self._pool_obj[self._excitation_dictionary[op_idx]][1].terms()[0][1]) - 1] += 1
                            break

                self._n_classical_params_lst.append(len(self._tops))

            if self._mmcc:
                mmcc_res = [res_coeffs[I] for I in self._mmcc_aux_excitation_indices]
                mmcc_res_sq_over_mpdenom = [np.real(np.conj(mmcc_res[I]) * mmcc_res[I] / self._mpdenom[I]) for I in range(len(mmcc_res))]
                if self._Egs is not None:
                    self._E_mmcc_mp.append(self._Egs + sum(mmcc_res_sq_over_mpdenom))
                    mmcc_res_sq_over_epstein_nesbet_denom = [np.real(np.conj(mmcc_res[I]) * mmcc_res[I] / (self._Egs - self._epstein_nesbet[I])) for I in range(len(mmcc_res))]
                    self._E_mmcc_en.append(self._Egs + sum(mmcc_res_sq_over_epstein_nesbet_denom))
                    #self._epstein_nesbet_like_denom = []
                    #for i in self._pool_obj.terms():
                    #    sq_op = i[1]
                    #    qc = qf.Computer(self._nqb)
                    #    qc.apply_circuit(self._Uprep)
                    #    qc.apply_operator(sq_op.jw_transform(self._qubit_excitations))
                    #    qc.apply_circuit(U)
                    #    self._epstein_nesbet_like_denom.append(qc.direct_op_exp_val(self._qb_ham))
                    #mmcc_res_sq_over_epstein_nesbet_like_denom = [np.real(np.conj(mmcc_res[I]) * mmcc_res[I] / (self._Egs - self._epstein_nesbet_like_denom[I])) for I in range(len(mmcc_res))]
                    #print(self._Egs + sum(mmcc_res_sq_over_epstein_nesbet_like_denom))
                #else:
                #    print(self._hf_energy + sum(mmcc_res_sq_over_mpdenom))
                #print('##########')

    def conv_status(self):
        if abs(self._curr_res_sq_norm) < abs(self._spqe_thresh * self._spqe_thresh):
            self._converged = True
            print("\n\n\n------------------------------------------------")
            print("SPQE macro-iterations converged!")
            print(f'||r|| = {np.sqrt(self._curr_res_sq_norm):8.6f}')
            print("------------------------------------------------")
            #self._final_result = self._results[-1]
        elif self._spqe_iter > self._spqe_maxiter:
            print("\n\n\n------------------------------------------------")
            print("Maximum number of SPQE macro-iterations reached!")
            print(f'Current value of ||r||: {np.sqrt(self._curr_res_sq_norm):8.6f}')
            print("------------------------------------------------")
            self._converged = True
        elif len(self._tops) == len(self._pool_obj):
            print("\n\n\n------------------------------------------------")
            print("Operator pool has been drained!")
            print("------------------------------------------------")
            self._converged = True
        else:
            self._converged = False

SPQE.jacobi_solver = optimizer.jacobi_solver
