import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from clases.hamiltoniano import Hamiltonian
from clases.metodos import Schrodinger

def main():
    Jt = 3.0
    gt = 1.5
    N = 2
    ham = Hamiltonian(N, Jt, gt)
    Hamr, Sz= ham.hamiltonian()
    ham.Psi_init()
    print(ham.psi_init)
    print(Hamr)

    metodos = Schrodinger( Hamr, N, ham.psi_init, Sz)

    metodos.vals_expect('rk4')

    metodos.vals_expect('exp')

if __name__ == "__main__":
    main()
