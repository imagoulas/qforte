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

import numpy as np
from itertools import combinations
from scipy.optimize import minimize

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
            optimizer='DIIS',
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

        for occupation in self._ref:
            if occupation:
                self._nbody_counts.append(0)

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
        idx = -1
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
                                       [len(betas) -i - 1 for i, x in enumerate(betas) if x]) == 0:
                    if idx == -1:
                        # Currently, the first determinant satisfying the required symmetry criteria is the
                        # HF determinant and is, thus, discarded
                        idx += 1
                        continue
                    excit = bin(ref ^ I).replace("0b", "")
                    if int(excit.count('1')/2) <= self._pool_type:
                        occ_idx = [int(i) for i,j in enumerate(reversed(excit[-nalpha-nbeta:])) if int(j) == 1]
                        unocc_idx = [int(i)+nalpha+nbeta for i,j in enumerate(reversed(excit[:-nalpha-nbeta])) if int(j) == 1]
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
                        idx += 1

        # excitation operator index : determinant id
        self._reversed_excitation_dictionary = {value: key for key, value in self._excitation_dictionary.items()}

        self.print_options_banner()

        self.build_orb_energies()
        spqe_iter = 0
        hit_maxiter = 0

        if(self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write(f"#{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')

        while not self._converged:

            print('\n\n -----> SPQE iteration ', spqe_iter, ' <-----\n')
            self.update_ansatz()

            if self._converged:
                break

            if(self._verbose):
                print('\ntoperators included from pool: \n', self._tops)
                print('\ntamplitudes for tops: \n', self._tamps)

            self.solve()

            if(self._verbose):
                print('\ntamplitudes for tops post solve: \n', np.real(self._tamps))

            if(self._print_summary_file):
                f.write(f'  {spqe_iter:7}    {self._energies[-1]:+15.9f}    {len(self._tamps):8}        {self._n_cnot_lst[-1]:10}        {sum(self._n_pauli_trm_measures_lst):12}\n')
            spqe_iter += 1

            if spqe_iter > self._spqe_maxiter-1:
                hit_maxiter = 1
                break

        if(self._print_summary_file):
            f.close()

        if hit_maxiter:
            self._Egs = self.get_final_energy(hit_max_spqe_iter=1)

        self._Egs = self.get_final_energy()

        print("\n\n")
        print("---> Final n-body excitation counts in SPQE ansatz <---")
        print("\n")
        print(f"{'Excitaion order':>20}{'Number of operators':>30}")
        print('---------------------------------------------------------')
        for l, nl in enumerate(self._nbody_counts):
            print(f"{l+1:12}              {nl:14}")

        print('\n\n')
        print(f"{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}")
        print('-------------------------------------------------------------------------------')

        for k, Ek in enumerate(self._energies):
            print(f' {k:7}    {Ek:+15.9f}    {self._n_classical_params_lst[k]:8}        {self._n_cnot_lst[k]:10}        {sum(self._n_pauli_trm_measures_lst[:k+1]):12}')

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
        if self._optimizer.lower() == 'diis':
            self.diis_solve()
        elif self._optimizer.lower() in ['nelder-mead', 'powell', 'cobyla', 'bfgs']:
            self.scipy_solve()
        else:
           raise NotImplementedError('Currently only DIIS, Nelder-Mead, and Powell solvers are implemented')


    def diis_solve(self):
        # draws heavy insiration from Daniel Smith's ccsd_diss.py code in psi4 numpy
        diis_dim = 0
        diis_max_dim = self._diis_max_dim
        t_diis = [copy.deepcopy(self._tamps)]
        e_diis = []
        Ek0 = self.energy_feval(self._tamps)

        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||')
        print('---------------------------------------------------------------------------------------------------')

        for k in range(1, self._opt_maxiter+1):
            t_old = copy.deepcopy(self._tamps)

            #do regular update
            r_k = self.get_residual_vector(self._tamps)
            r_k = self.get_res_over_mpdenom(r_k, self._shift)

            self._tamps = list(np.add(self._tamps, r_k))

            Ek = self.energy_feval(self._tamps)
            dE = Ek - Ek0
            Ek0 = Ek

            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.10f}')

            if(self._res_vec_norm < self._opt_thresh):
                self._results.append('Fake result string')
                self._final_result = 'nothing'
                self._Egs = Ek
                break

            t_diis.append(copy.deepcopy(self._tamps))
            e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

            if(k >= 1 and diis_max_dim >= 2):

                if len(t_diis) > diis_max_dim:
                    del t_diis[0]
                    del e_diis[0]

                diis_dim = len(t_diis) - 1

                #consturct diis B matrix (following Crawford Group github tutorial)
                B = np.ones((diis_dim+1, diis_dim+1)) * -1
                bsol = np.zeros(diis_dim+1)
                B[-1, -1] = 0.0
                bsol[-1] = -1.0

                for i in range(len(e_diis)):
                    for j in range(i, len(e_diis)):
                        B[i,j] = np.dot(np.real(e_diis[i]), np.real(e_diis[j]))
                        if(i!=j):
                            B[j,i] = B[i,j]

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
                x = np.linalg.solve(B, bsol)

                t_new = np.zeros(( len(self._tamps) ))
                for l in range(diis_dim):
                    temp_ary = x[l] * np.asarray(t_diis[l+1])
                    t_new = np.add(t_new, temp_ary)

                self._tamps = copy.deepcopy(list(np.real(t_new)))

        self._Egs = Ek
        self._energies.append(Ek)
        self._n_pauli_measures_k += self._Nl*k * (2*len(self._tamps) + 1)
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
        self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())

    def scipy_solve(self):

        # Construct arguments to hand to the minimizer.
        opts = {}

        # Options common to all minimization algorithms
        opts['disp'] = True
        opts['maxiter'] = self._opt_maxiter

        # Optimizer-specific options
        if self._optimizer.lower() == 'nelder-mead':
            opts['fatol'] = self._opt_thresh
        if self._optimizer.lower() in ['powell']:
            opts['ftol'] = self._opt_thresh

        x0 = copy.deepcopy(self._tamps)
        self._prev_energy = self.energy_feval(x0)
        self._k_counter = 0

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

            if self._optimizer.lower() == 'diis':
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

        res_coeffs = qc_res.get_coeff_vec()

        # build different res_sq list using M_omega
        if(self._M_omega != 'inf'):
            res_sq_tmp = [ np.real(np.conj(res_coeffs[I]) * res_coeffs[I]) for I in range(len(res_coeffs))]

            # Nmu_lst => [ det1, det2, det3, ... det_M_omega]
            det_lst = np.random.choice(len(res_coeffs), self._M_omega, p=res_sq_tmp)

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
            res_sq = [( np.real(np.conj(res_coeffs[I]) * res_coeffs[I]), I) for I in self._excitation_indices]
            res_sq.sort()
            self._curr_res_sq_norm = sum(rmu_sq[0] for rmu_sq in res_sq) / (self._dt * self._dt)

            print('  \n--> Begin selection opt with residual magnitudes |r_mu|:')
            print('  Initial guess energy: ', round(init_gues_energy,10))
            print(f'  Norm of res vec:      {np.sqrt(self._curr_res_sq_norm):14.12f}')

            self.conv_status()

            if not self._converged:
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
                        res_sq_sum += rmu_sq / (self._dt * self._dt)
                        if res_sq_sum > (self._spqe_thresh * self._spqe_thresh):
                            if(self._verbose):
                                print(f"  {self._excitation_dictionary[op_idx]:10}                  {np.real(rmu_sq)/(self._dt * self._dt):14.12f}"
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
                        print(f"  {self._excitation_dictionary[op_idx]:10}                  {np.real(rmu_sq)/(self._dt * self._dt):14.12f}")
                        if self._excitation_dictionary[op_idx] not in self._tops:
                            print('Adding this operator to ansatz')
                            self._tops.insert(0, self._excitation_dictionary[op_idx])
                            self._tamps.insert(0, 0.0)
                            self._nbody_counts[len(self._pool_obj[self._excitation_dictionary[op_idx]][1].terms()[0][1]) - 1] += 1
                            break

                self._n_classical_params_lst.append(len(self._tops))

    def conv_status(self):
        if abs(self._curr_res_sq_norm) < abs(self._spqe_thresh * self._spqe_thresh):
            self._converged = True
            self._final_energy = self._energies[-1]
            #self._final_result = self._results[-1]
        else:
            self._converged = False

    def get_final_energy(self, hit_max_spqe_iter=0):
        """
        Parameters
        ----------
        hit_max_spqe_iter : bool
            Wether or not to use the SPQE has already hit the maximum
            number of iterations.
        """
        if hit_max_spqe_iter:
            print("\nSPQE at maximum number of iterations!")
            self._final_energy = self._energies[-1]
        else:
            return self._final_energy
