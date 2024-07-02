# Se Importan librerías
import numpy as np
import qutip as qt  # Funcion "tensor" para producto tensorial entre elementos de un arreglo
import time         # Funcion "timeit" para medir el tiempo de ejecucion


# Esta función es la ecuación de schrodinger
def schrodinger(ham, psi):

    return -1.0j*(np.dot(ham, psi))



# Esta rutina devuelve una matrix 2^Nx2^N que corresponde al Hamiltoniano del modelo de Ising para N espines
def hamiltonian(J, g, N):
    
    sx = qt.sigmax()    # Matriz de Pauli en X
    sz = qt.sigmaz()    # Matriz de Pauli en Z
    iden = qt.qeye(2)   # Matriz Identidad de tamaño 2x2

    sigma_z = [0] * N   # Arreglos donde se van a guardar los sigma^z_i y los sigma^x_i
    sigma_x = [0] * N   # Se crean los arreglos de esta forma para evitar hacer "append"
    
    for i in range(N):
                           
        sigma_zi = [0] * N  # Arreglos donde se guardan las matrices 2x2   
        sigma_xi = [0] * N  # a las que se les va a aplicar producto tensorial

        for j in range(i):

            # Se guardan las matrices identidad que van antes de la i-ésima posición
            sigma_zi[j] = (iden)  
            sigma_xi[j] = (iden)
        
        # Se guarda la matriz de Pauli en la i-ésima posición
        sigma_zi[i] = (sz)
        sigma_xi[i] = (sx)

        for k in range(i+1,N):
            
            # Se guardan las matrices identidad que van después de la i-ésima posición         
            sigma_zi[k] = (iden) 
            sigma_xi[k] = (iden)
        
        # Producto tensorial empezando en la primera posición hasta la
        # última posición de los elementos del vector sigma_zi
        sigma_z[i] = (qt.tensor(sigma_zi)).full()
        sigma_x[i] = (qt.tensor(sigma_xi)).full()
    
    h_first = 0.0  # Primer término del hamiltoniano
    h_second = 0.0 # Segundo término del hamiltoniano

    # Se realiza el calculo de los dos términos del Hamiltoniano, tomando en cuante condiciones de fronter periodicas
    for i in range(N):
        h_first += J*np.dot(sigma_z[i], sigma_z[(i+1)%N])

    for i in range(N):
        h_second += g*np.array(sigma_x[i])

    # Se devuelve el hamiltoniano y sigma_z(Sz)
    return (h_first + h_second), sigma_z


# Esta función realiza el método de Runge-Kutta 4 para 1 iteración

def rk4(func, ham, y_n, h):

    k_1 = func(ham, y_n) 
    k_2 = func(ham, y_n + (h/2) * k_1)
    k_3 = func(ham, y_n + (h/2) * k_2)
    k_4 = func(ham, y_n + h * k_3)

    return y_n + (h/6)*(k_1+2*k_2+2*k_3+k_4)


# Empezamos por definir el numero de espines
N = 8 

# Luego definimos el estado inicial
psi_init = np.zeros(2**N)
psi_init[1] = 1.0


# Ahora definimos los parametros del Hamiltoniano y lo construimos
Jt = 3.0
gt = 1.5
Hamiltonian, Sz = hamiltonian(Jt, gt, N)

# Aquí se discretiza el tiempo y se define el timestep
times = np.linspace(0.0, 25.0, 2001)
h = times[1] - times[0]

# Ahora se reserva la memoria donde se van a guardar los valores de expectacion de cada espin para cada valor de tiempo
obs = np.zeros((N,times.size), dtype=complex)



# Se mide el tiempo en este punto del codigo
start = time.time()  

# Aquí se diagonaliza el Hamiltoniano con numpy y se obtienen los valores de la diagonal y la matriz de transformacion
eigen_val, eigen_vec = np.linalg.eigh(Hamiltonian)

# Ahora se transforma tanto los Sz's como el estaado inicial para que todo este en la misma base
Sz_prime = [ np.dot( np.dot(eigen_vec.transpose(), Sz[i]), eigen_vec) for i in range(N)]

psi_init_prime = psi_init.dot(eigen_vec)


# Ahora se evalua iterativamente la exponencial

psi_0 = psi_init_prime
for tt in range(times.size):
    
    # Se calculan los valores de expectación de los Sz's utilizando psi_0
    valor_exp = [ np.dot( np.dot(psi_0.conjugate().transpose(), Sz_prime[i]) , psi_0) for i in range(N)]

    # Se asignan los valores de expectacion al espin y tiempo correspondiente de obs
    for i in range(N):
        obs[i][tt] = valor_exp[i]

    # Se evalua la exponencial obteniendo asi el propagador evaluado en el tiempo tt*h
    propagador = np.exp((-1.0j*eigen_val)*(tt*h))
    
    # Se multiplica el estado inicial por el progador y se obtiene el estado en el tiempo tt*h
    # Como el propagador es una matriz diagonal, multiplicar por un vector es lo mismo que multiplicar entrada por entrada la diagonal con el vector
    psi_n = np.array([ propagador[i] * psi_init_prime[i] for i in range(2**N)] )
    
    # Ahora creamos un shallow copy de psi_n en psi_0
    # De esta manera, en la siguiente iteración, el estado
    # de esta iteración se convierte en el inicial de la siguiente iteración
    psi_0 = psi_n


# Se mide el tiempo en este punto del codigo
end = time.time()

# Se imprime el tiempo de ejecucion
print(end - start)
