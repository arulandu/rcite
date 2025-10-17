import numpy as np
import scipy
from scipy import stats
import qiskit
from functools import reduce

def pauli_str_to_unitary(pstr):
    def char_to_pauli(c):
        if c == "I":
            return np.eye(2)
        elif c == "X":
            return np.array([[0, 1], [1, 0]])
        elif c == "Y":
            return np.array([[0, -1j], [1j, 0]])
        elif c == "Z":
            return np.array([[1, 0], [0, -1]])
        else:
            raise ValueError(f"Invalid Pauli character: {c}")
    
    return reduce(np.kron, [char_to_pauli(c) for c in pstr], np.array([[1]]))

def spin_chain_H(n, J=1, g=1, type="transverse"):
    if type == "transverse":
        H = np.zeros((2**n, 2**n))
        for i in range(n):
            if i < n-1:
                pstr = list("I"*n)
                pstr[i] = pstr[i+1] = "Z"
                H += pauli_str_to_unitary(pstr)
            
            pstr = list("I"*n)
            pstr[i] = "X"
            H += g*pauli_str_to_unitary(pstr)

        H *= -J
    return H

def rand_prod_state(n):
    def rand_state():
        return qiskit.quantum_info.random_statevector(2)

    return reduce(np.kron, [rand_state() for _ in range(n)], np.array(1))

def outer(state):
    psi = state.reshape(-1, 1)
    return psi @ psi.conj().T

gcauchy_pdf = lambda u: 1/np.pi * 1/(1+(u/np.sqrt(2))**4)

def sample_gcauchy(M=1):
    cs = []
    thresh = (1+np.sqrt(5))/2 # sup u of pdf(u)/cauchypdf(u) = golden ratio
    while len(cs) < M:
        c = scipy.stats.cauchy.rvs()
        u = scipy.stats.uniform.rvs()
        h = (1+c**2)/(1+(c/np.sqrt(2))**4)
        if u < h/thresh:
            cs.append(c)

    return np.array(cs) if M > 1 else cs[0]

class Hamiltonian:
    def __init__(self, H):
        self.H = H
        self.n = H.shape[0]

        # Eigendecomposition of Hamiltonian
        eigenvals, eigenvecs = np.linalg.eigh(self.H)
        self.U = eigenvecs  # Unitary that diagonalizes H
        self.D = np.diag(eigenvals)  # Diagonal matrix of eigenvalues
        self.lam = eigenvals
        
        self.ground = eigenvecs[:, 0]
    
    def ITE(self, H, beta, method="fast"):
        if method == "direct":
            return scipy.linalg.expm(-beta * H)
        elif method == "fast":
            return self.U @ np.diag(np.exp(-beta * self.lam)) @ self.U.conj().T
        else:
            raise ValueError(f"Invalid method: {method}")

    def RTE(self, H, t, method="fast"):
        if method == "direct":
            return scipy.linalg.expm(-1j * H * t)   
        elif method == "fast":
            return self.U @ np.diag(np.exp(-1j * self.lam * t)) @ self.U.conj().T
        else:
            raise ValueError(f"Invalid method: {method}")
    
    def iITE(self, state, beta, method="fast"):
        psi_ = self.ITE(self.H, beta, method=method) @ state
        psi = psi_ / np.linalg.norm(psi_)
        return psi

    def iITE_rand_iRTE_fixed(self, state, beta, M=1000, sample=sample_gcauchy, pdf=gcauchy_pdf, method="fast", **kwargs):
        """
        Fixed version of iITE_rand_iRTE that prevents NaN values in the integrate method
        """
        psi = self.iITE(state, beta, method=("direct" if method == "direct" else "fast"))

        if method == "direct":
            samples = np.array([sample() for _ in range(M)])
            states = [self.RTE(self.H, t) @ psi for t in samples]
            rho = sum([outer(s) for s in states]) / M
        elif method == "fast":
            samples = np.array([sample() for _ in range(M)])
            c = self.U.conj().T @ psi
            phases = np.exp(-1j * np.outer(self.lam, samples))
            c_t = c[:, None] * phases
            psi_t = self.U @ c_t
            rho = psi_t @ psi_t.conj().T / M
        elif method == "integrate": 
            # Fixed integrate method to prevent NaN values
            rho_0 = outer(psi)
            rho_U = self.U.conj().T @ rho_0 @ self.U
            
            # Use finite integration limits to avoid numerical issues
            # For generalized Cauchy, most weight is within ~10-20 standard deviations
            max_t = 100.0  # Finite integration limit
            min_t = -max_t
            
            def integrand(t):
                # Handle scalar t input
                if np.isscalar(t):
                    t = np.array([t])
                
                # Compute phase factors with numerical stability
                # Use clip to prevent overflow in exponential
                t_clipped = np.clip(t, -1e6, 1e6)
                phase_neg = np.exp(-1j * np.outer(self.lam, t_clipped))
                phase_pos = np.exp(1j * np.outer(self.lam, t_clipped))
                
                # Compute PDF values safely
                pdf_vals = np.array([pdf(ti) if not np.isnan(pdf(ti)) else 0.0 for ti in t])
                
                # Initialize result array
                result = np.zeros((len(t), self.n, self.n), dtype=complex)
                
                for i, (ti, pdf_val) in enumerate(zip(t, pdf_vals)):
                    if pdf_val == 0.0:
                        continue
                    
                    # Compute the matrix product step by step for numerical stability
                    diag_neg = np.diag(phase_neg[:, i])
                    diag_pos = np.diag(phase_pos[:, i])
                    
                    # U @ diag_neg @ rho_U @ diag_pos @ U.conj().T
                    temp1 = diag_neg @ rho_U
                    temp2 = temp1 @ diag_pos
                    temp3 = self.U @ temp2
                    result[i] = pdf_val * (temp3 @ self.U.conj().T)
                
                return result
            
            # Use finite integration limits and better error handling
            try:
                rho = scipy.integrate.quad_vec(integrand, min_t, max_t, **kwargs)[0]
                
                # Check for NaN or inf values and handle them
                if np.any(np.isnan(rho)) or np.any(np.isinf(rho)):
                    print("Warning: NaN or inf detected in integration result. Using fast method as fallback.")
                    # Fallback to fast method
                    samples = np.array([sample() for _ in range(M)])
                    c = self.U.conj().T @ psi
                    phases = np.exp(-1j * np.outer(self.lam, samples))
                    c_t = c[:, None] * phases
                    psi_t = self.U @ c_t
                    rho = psi_t @ psi_t.conj().T / M
                    
            except Exception as e:
                print(f"Integration failed: {e}. Using fast method as fallback.")
                # Fallback to fast method
                samples = np.array([sample() for _ in range(M)])
                c = self.U.conj().T @ psi
                phases = np.exp(-1j * np.outer(self.lam, samples))
                c_t = c[:, None] * phases
                psi_t = self.U @ c_t
                rho = psi_t @ psi_t.conj().T / M
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return rho

# Test the fixed method
if __name__ == "__main__":
    # Test parameters
    n = 4
    g = 1.4
    J = 1.0
    H = spin_chain_H(n, J=J, g=g)
    hamil = Hamiltonian(H)
    psi0 = rand_prod_state(n)
    beta = 1
    
    print("Testing fixed integrate method...")
    rho_fixed = hamil.iITE_rand_iRTE_fixed(psi0, beta, method="integrate", epsrel=1e-4)
    print("Fixed method result shape:", rho_fixed.shape)
    print("Contains NaN:", np.any(np.isnan(rho_fixed)))
    print("Contains Inf:", np.any(np.isinf(rho_fixed)))
    print("Trace:", np.trace(rho_fixed))
    print("Trace distance from identity:", np.sum(np.abs(np.linalg.svdvals(rho_fixed - np.eye(rho_fixed.shape[0])))) / 2) 