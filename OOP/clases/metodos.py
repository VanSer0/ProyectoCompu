import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


class Schrodinger:
    def __init__(self, ham, N, psi, Sz):
        self.ham = ham
        self.N = int(N)
        self.method = ""
        self.psi = psi
        self.Sz = Sz

    def schrodinger(self, ham, psi):

        return -1.0j*(np.dot(ham, psi))

    def rk4(self):
        k_1 = self.schrodinger(self.ham, self.psi_0)
        k_2 = self.schrodinger(self.ham, self.psi_0 + (self.h/2) * k_1)
        k_3 = self.schrodinger(self.ham, self.psi_0 + (self.h/2) * k_2)
        k_4 = self.schrodinger(self.ham, self.psi_0 + self.h * k_3)

        return self.psi_0 + (self.h/6)*(k_1+2*k_2+2*k_3+k_4)

    def exponencial(self):
        self.eigen_val, self.eigen_vec = np.linalg.eigh(self.ham)
        self.Sz_prime = [ np.dot( np.dot(self.eigen_vec.transpose(), self.Sz[i]), self.eigen_vec) for i in range(self.N)]
        self.psi_init_prime = self.psi.dot(self.eigen_vec)

    def times_discreto(self):
        self.times = np.linspace(0.0, 25.0, 2001)
        self.h = self.times[1] - self.times[0]
        self.obs = np.zeros((self.N,self.times.size), dtype=complex)

    def vals_expect(self, k):
        self.method=k
        self.times_discreto()

        if self.method=='rk4':
            self.psi_0 = self.psi

        elif self.method == 'exp':
            self.exponencial()
            self.psi_0 =  self.psi_init_prime
            self.Sz = self.Sz_prime
        else:
            print('error')

        for tt in range(self.times.size):

            self.valor_exp = [ np.dot( np.dot(self.psi_0.conjugate().transpose(), self.Sz[i]) , self.psi_0) for i in range(self.N)]

            for i in range(self.N):
                self.obs[i][tt] = self.valor_exp[i]

            if self.method == 'rk4':
                psi_n = self.rk4()
            else:
                propagador = np.exp((-1.0j*self.eigen_val)*(tt*self.h))
                psi_n = np.array([ propagador[i] * self.psi_init_prime[i] for i in range(2**self.N)] )



            self.psi_0=psi_n
        self.graficar()


    def graficar(self):
        plt.figure(figsize=(14,7))

        ls=['-','--','-.',':']

        for i in range(self.N):
            plt.plot(self.times, np.real(self.obs[i]),
                     linestyle=ls[i%4],
                     linewidth=4,
                     alpha=0.8,
                     label='Espín: {0}'.format(i))

        plt.title(self.method)
        plt.legend()
        plt.show()

