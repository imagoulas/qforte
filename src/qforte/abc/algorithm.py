"""
algorithm.py
====================================
The abstract base class inherited by all algorithm subclasses.
"""

from abc import ABC, abstractmethod
from qforte.utils.state_prep import *

class Algorithm(ABC):
    """
    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    _nqb : int
        The number of qubits the calculation empolys.

    _qb_ham : QuantumOperator
        The operator to be measured (usually the Hamiltonian), mapped to a
        qubit representation.

    _fast : bool
        Whether or not to use a faster version of the algorithm that bypasses
        measurment (unphysical for quantum computer). Most algorithms only
        have a fast implentation.

    _trotter_order : int
        The Trotter order to use for exponentiated operators.
        (exact in the infinte limit).

    _trotter_number : int
        The Trotter number (or the number of trotter steps)
        to use for exponentiated operators.
        (exact in the infinte limit).

    _Egs : float
        The final ground state energy value.

    _Umaxdepth : QuantumCircuit
        The deepest circuit used during any part of the algorithm.

    _n_classical_params : int
        The number of classical parameters used by the algorithm.

    _n_cnot : int
        The number of controlled-not (CNOT) opperations used in the (deepest)
        quantum circuit (_Umaxdepth).

    _n_pauli_trm_measures : int
        The number of pauli terms (Hermitian products of Pauli X, Y, and/or Z gates)
        mesaured over the entire algorithm.

    Methods
    -------
    print_options_banner()
        Prints the run options used for algorithm.

    print_summary_banner()
        Prints a summary of the post-run information.

    run()
        Execute the algorithm.

    run_realistic()
        Executes the algorithm using only operations physically possable for
        quantum hardware. Not implented for most algorithms.

    verify_run()
        Verifies that the abstract sub-class(es) define the required attributes.

    get_gs_energy()
        Returns the final ground state energy

    def get_Umaxdepth()
        Returns the deepest circuit used during any part of the
        algorithm (_Umaxdepth).

    def verify_required_attributes()
        Verifies that the concrete sub-class(es) define the required attributes.

    """

    def __init__(self,
                 system,
                 reference,
                 trial_state_type='reference',
                 trotter_order=1,
                 trotter_number=1,
                 fast=True,
                 verbose=False,
                 print_summary_file=False):

        self._sys = system
        self._ref = reference
        self._nqb = len(reference)
        self._trial_state_type = trial_state_type
        self._Uprep = build_Uprep(reference, trial_state_type)
        # TODO (Nick): change Molecule.get_hamiltonian() to Molecule.get_qb_hamiltonain()
        self._qb_ham = system.get_hamiltonian()
        if hasattr(system, '_hf_energy'):
            self._hf_energy = system.get_hf_energy()
        else:
            self._hf_energy = 0.0

        self._Nl = len(self._qb_ham.terms())
        self._trotter_order = trotter_order
        self._trotter_number = trotter_number
        self._fast = fast
        self._verbose = verbose
        self._print_summary_file = print_summary_file

        self._noise_factor = 0.0

        # Required attributes, to be defined in concrete class.
        self._Egs = None
        self._Umaxdepth = None
        self._n_classical_params = None
        self._n_cnot = None
        self._n_pauli_trm_measures = None


    @abstractmethod
    def print_options_banner(self):
        pass

    @abstractmethod
    def print_summary_banner(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def run_realistic(self):
        pass

    @abstractmethod
    def verify_run(self):
        pass

    def get_gs_energy(self):
        return self._Egs

    def get_Umaxdepth(self):
        pass

    def get_tot_measurements(self):
        pass

    def get_tot_state_preparations(self):
        pass

    def verify_required_attributes(self):
        if self._Egs is None:
            raise NotImplementedError('Concrete Algorithm class must define self._Egs attribute.')

#         if self._Umaxdepth is None:
#             raise NotImplementedError('Concrete Algorithm class must define self._Umaxdepth attribute.')

        if self._n_classical_params is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_classical_params attribute.')

        if self._n_cnot is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_cnot attribute.')

        if self._n_pauli_trm_measures is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_pauli_trm_measures attribute.')
