import qforte

import copy
import numpy as np

def diis(diis_max_dim, t_diis, e_diis):
    """This function implements the direct inversion of iterative subspace
    (DIIS) convergence accelerator. Draws heavy insiration from Daniel
    Smith's ccsd_diss.py code in psi4 numpy
    """

    if len(t_diis) > diis_max_dim:
        del t_diis[0]
        del e_diis[0]

    diis_dim = len(t_diis) - 1

    # Construct diis B matrix (following Crawford Group github tutorial)
    B = np.ones((diis_dim+1, diis_dim+1)) * -1
    bsol = np.zeros(diis_dim+1)

    B[-1, -1] = 0.0
    bsol[-1] = -1.0
    for i, ei in enumerate(e_diis):
        for j, ej in enumerate(e_diis):
            B[i,j] = np.dot(np.real(ei), np.real(ej))

    B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

    x = np.linalg.solve(B, bsol)

    t_new = np.zeros(( len(t_diis[0]) ))
    for l in range(diis_dim):
        temp_ary = x[l] * np.asarray(t_diis[l+1])
        t_new = np.add(t_new, temp_ary)

    return copy.deepcopy(list(np.real(t_new)))

def jacobi_solver(self):
    """
    """

    t_diis = [copy.deepcopy(self._tamps)]
    e_diis = []
    Ek0 = self.energy_feval(self._tamps)

    print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||')
    print('---------------------------------------------------------------------------------------------------', flush=True)

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
            self._Egs = Ek
            break

        t_diis.append(copy.deepcopy(self._tamps))
        e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

        if(k >= 1 and self._diis_max_dim >= 2):
            self._tamps = diis(self._diis_max_dim, t_diis, e_diis)

    self._Egs = Ek
    if k == self._opt_maxiter:
        print("\nMaximum number of Jacobi iterations reached!")
    if hasattr(self, '_energies'):
        self._energies.append(Ek)
    if hasattr(self, '_n_classical_params'):
        self._n_classical_params = len(self._tamps)
    if hasattr(self, '_n_pauli_measures_k'):
        self._n_pauli_measures_k += self._Nl*k * (2*len(self._tamps) + 1)
    if hasattr(self, '_n_pauli_trm_measures'):
        self._n_pauli_trm_measures += 2*self._Nl*k*len(self._tamps) + self._Nl*k
    if hasattr(self, '_n_pauli_trm_measures_lst'):
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
    if hasattr(self, '_n_cnot'):
        self._n_cnot = self.build_Uvqc().get_num_cnots()
    if hasattr(self, '_n_cnot_lst'):
        self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())

