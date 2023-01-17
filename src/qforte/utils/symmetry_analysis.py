import qforte as qf
import numpy as np

def symmetry_analysis(self):

    # Generate given quantum state
    qc = qf.Computer(self._nqb)
    qc.apply_circuit(self._Uprep)
    qc.apply_circuit(self.ansatz_circuit())

    # Compute <N> and <N^2> - <N>^2, where N is the total number operator
    N = qf.QubitOperator()
    circ_const = qf.Circuit()
    N.add_term(self._nqb / 2, circ_const)
    for qubit in range(self._nqb):
        circ_Z = qf.Circuit()
        circ_Z.add_gate(qf.gate('Z', qubit))
        N.add_term(-0.5, circ_Z)
    N_sqrd = qf.QubitOperator()
    N_sqrd.add_op(N)
    N_sqrd.operator_product(N, True, True)
    N_exp_val = qc.direct_op_exp_val(N)
    N_var = qc.direct_op_exp_val(N_sqrd) - N_exp_val**2
    if np.imag(N_exp_val) != 0 or np.imag(N_var) != 0:
        raise ValueError('The expectation value and variance of the particle number should be real!')
    N_exp_val = np.real(N_exp_val)
    N_var = np.real(N_var)

    # Compute <Sz> and <Sz^2> - <Sz>^2, where Sz is projection of the total spin on the z axis
    Sz = qf.total_spin_z(self._nqb)
    Sz_sqrd = qf.QubitOperator()
    Sz_sqrd.add_op(Sz)
    Sz_sqrd.operator_product(Sz, True, True)
    Sz_exp_val = qc.direct_op_exp_val(Sz)
    Sz_var = qc.direct_op_exp_val(Sz_sqrd) - Sz_exp_val**2
    if np.imag(Sz_exp_val) != 0 or np.imag(Sz_var) != 0:
        raise ValueError('The expectation value and variance of Sz should be real!')
    Sz_exp_val = np.real(Sz_exp_val)
    Sz_var = np.real(Sz_var)

    # Compute <S^2> and <S^4> - <S^2>^2, where S is the total spin
    S_sqrd = qf.total_spin_squared(self._nqb)
    S_sqrd_sqrd = qf.QubitOperator()
    S_sqrd_sqrd.add_op(S_sqrd)
    S_sqrd_sqrd.operator_product(S_sqrd, True, True)
    S_sqrd_exp_val = qc.direct_op_exp_val(S_sqrd)
    S_sqrd_var = qc.direct_op_exp_val(S_sqrd_sqrd) - S_sqrd_exp_val**2
    if np.imag(S_sqrd_exp_val) !=0 or np.imag(S_sqrd_var) !=0:
        raise ValueError('The expectation value and variance of S^2 should be real!')
    S_sqrd_exp_val = np.real(S_sqrd_exp_val)
    S_sqrd_var = np.real(S_sqrd_var)

    # Compute the weight of determinants with correct irrep
    weight = 0
    for det in range(1<<self._nqb):
        occ = []
        for i in range(self._nqb):
            if (1<<i)&det !=0:
                occ.append(i)
        if qf.sq_op_find_symmetry(self._sys.orb_irreps_to_int, occ, []) == 0:
            coeff = qc.get_coeff_vec()[det]
            weight += coeff * np.conjugate(coeff)
    weight = np.real(weight)


    print('\n\n           Symmetry Analysis')
    print('----------------------------------------')
    print('<N>:                      ', f'{N_exp_val:+.10f}')
    print('<N^2> - <N>^2:            ', f'{N_var:+.10f}')
    print('<Sz>:                     ', f'{Sz_exp_val:+.10f}')
    print('<Sz^2> - <Sz>^2:          ', f'{Sz_var:+.10f}')
    print('<S^2>:                    ', f'{S_sqrd_exp_val:+.10f}')
    print('<S^4> - <S^2>^2:          ', f'{S_sqrd_var:+.10f}')
    print('Totally symmetric weight: ', f'{weight:+.10f}')
