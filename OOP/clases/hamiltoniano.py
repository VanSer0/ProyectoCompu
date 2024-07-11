import numpy as np
import qutip as qt
class Hamiltonian:
    def __init__(self, N, J, g):
        self.N = N
        self.J = J
        self.g = g

        self.sx = qt.sigmax()
        self.sz = qt.sigmaz()
        self.iden = qt.qeye(2)

        self.psi_init = np.zeros(2**int(self.N))


    def hamiltonian(self):
        self.sigma_z = [0] * int(self.N)
        self.sigma_x = [0] * int(self.N)

        for i in range(int(self.N)):

            sigma_zi = [0] * int(self.N)
            sigma_xi = [0] * int(self.N)

            for j in range(i):

                sigma_zi[j] = (self.iden)
                sigma_xi[j] = (self.iden)

            sigma_zi[i] = (self.sz)
            sigma_xi[i] = (self.sx)

            for k in range(i+1,int(self.N)):

                sigma_zi[k] = (self.iden)
                sigma_xi[k] = (self.iden)

            self.sigma_z[i] = (qt.tensor(sigma_zi)).full()
            self.sigma_x[i] = (qt.tensor(sigma_xi)).full()


        self.h_first = 0.0
        self.h_second = 0.0

        for i in range(int(self.N)):
            self.h_first += self.J*np.dot(self.sigma_z[i], self.sigma_z[(i+1)%int(self.N)])

        for i in range(int(self.N)):

            self.h_second += self.g*np.array(self.sigma_x[i])


        return (self.h_first + self.h_second), self.sigma_z

    def Psi_init(self):
        self.psi_init[1] = 1.0 ##Podemos incluir que el inicial se crea de manera aleatoria, con un random entre 0 y N-1.
