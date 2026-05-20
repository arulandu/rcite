from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.stats
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


@dataclass
class ExperimentConfig:
    n: int = 4
    J: float = 1.0
    g: float = 1.4
    beta_min: float = 0.5
    beta_max: float = 5.0
    beta_num: int = 10
    spin_index: int = 2
    M_exact: int = 500
    M_circuit: int = 30
    shots: int = 2000
    trotter_steps: int = 10
    trotter_order: int = 2
    noise_levels: tuple[float, ...] = (0.0, 1e-4, 1e-3)
    target_trace_distance: float = 0.05
    seed: int = 7


def pauli(c: str) -> np.ndarray:
    return {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }[c]


def pauli_mat(pstr: str) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for c in pstr:
        out = np.kron(out, pauli(c))
    return out


def outer(v: np.ndarray) -> np.ndarray:
    v = v.reshape(-1, 1)
    return v @ v.conj().T


def expect(state: np.ndarray, H: np.ndarray) -> float:
    val = state.conj().T @ H @ state if state.ndim == 1 else np.trace(state @ H)
    return float(np.real_if_close(val))


def trdist(rho: np.ndarray, sigma: np.ndarray) -> float:
    return 0.5 * np.sum(np.abs(np.linalg.svd(rho - sigma, compute_uv=False)))


def spin_cov_xx(rho: np.ndarray, i: int) -> float:
    n = int(np.log2(rho.shape[0]))
    p = list("I" * n)
    p[i] = "X"
    xi = expect(rho, pauli_mat("".join(p)))
    p[i + 1] = "X"
    xixi1 = expect(rho, pauli_mat("".join(p)))
    p[i] = "I"
    xi1 = expect(rho, pauli_mat("".join(p)))
    return xixi1 - xi * xi1


def build_tfim_terms(n: int, J: float, g: float) -> list[tuple[str, float]]:
    terms: list[tuple[str, float]] = []
    for i in range(n - 1):
        p = list("I" * n)
        p[i] = p[i + 1] = "Z"
        terms.append(("".join(p), -J))
    for i in range(n):
        p = list("I" * n)
        p[i] = "X"
        terms.append(("".join(p), -J * g))
    return terms


def build_hamiltonian_matrix(terms: list[tuple[str, float]]) -> np.ndarray:
    n = len(terms[0][0])
    H = np.zeros((2**n, 2**n), dtype=complex)
    for p, w in terms:
        H += w * Operator.from_label(p[::-1]).data
    return H


def trotter_schedule(
    terms: list[tuple[str, float]], alpha: float, steps: int, order: int
) -> list[tuple[str, float]]:
    def rec(x: float, o: int) -> list[tuple[str, float]]:
        if o == 1:
            return [(p, x * w) for p, w in terms]
        if o == 2:
            fwd = [(p, 0.5 * x * w) for p, w in terms]
            return fwd + list(reversed(fwd))
        p = 1.0 / (4.0 - 4.0 ** (1.0 / (o - 1)))
        return rec(p * x, o - 2) + rec((1 - 2 * p) * x, o - 2) + rec(p * x, o - 2)

    return rec(alpha / steps, order) * steps


class ExactSim:
    def __init__(
        self,
        H: np.ndarray,
        terms: list[tuple[str, float]] | None = None,
        trotter_steps: int = 0,
        trotter_order: int = 2,
    ) -> None:
        self.H = H
        self.evals, self.evecs = np.linalg.eigh(H)
        self.ground = self.evecs[:, 0]
        self._use_trotter = trotter_steps > 0 and terms is not None
        self.terms = terms or []
        self.steps = trotter_steps
        self.order = trotter_order
        if self._use_trotter:
            self._term_map = {p: i for i, (p, _) in enumerate(self.terms)}
            self._term_eigs = []
            for p, _ in self.terms:
                lam, U = np.linalg.eigh(pauli_mat(p))
                self._term_eigs.append((lam, U))

    def _term_exp(self, idx: int, coeff: float) -> np.ndarray:
        lam, U = self._term_eigs[idx]
        return U @ np.diag(np.exp(lam * coeff)) @ U.conj().T

    def _ite_op(self, beta: float) -> np.ndarray:
        if not self._use_trotter:
            return self.evecs @ np.diag(np.exp(-beta * self.evals)) @ self.evecs.conj().T
        sched = trotter_schedule(self.terms, -beta, self.steps, self.order)
        op = np.eye(self.H.shape[0], dtype=complex)
        for p, c in sched:
            op = self._term_exp(self._term_map[p], c) @ op
        return op

    def ite(self, psi0: np.ndarray, beta: float) -> np.ndarray:
        psi = self._ite_op(beta) @ psi0
        return psi / np.linalg.norm(psi)

    def rte(self, psi: np.ndarray, t: float) -> np.ndarray:
        op = self.evecs @ np.diag(np.exp(-1j * t * self.evals)) @ self.evecs.conj().T
        return op @ psi

    def rite(self, psi0: np.ndarray, beta: float, M: int, sample_fn: Callable[[], float]) -> np.ndarray:
        psi = self.ite(psi0, beta)
        ts = np.array([sample_fn() for _ in range(M)])
        c = self.evecs.conj().T @ psi
        phases = np.exp(-1j * np.outer(self.evals, ts))
        states = self.evecs @ (c[:, None] * phases)
        return states @ states.conj().T / M


def gcauchy_sample(beta: float, rng: np.random.Generator) -> float:
    thresh = 0.5 * (1 + np.sqrt(5))
    while True:
        c = scipy.stats.cauchy.rvs(random_state=rng)
        u = scipy.stats.uniform.rvs(random_state=rng)
        if u < (1 + c**2) / (thresh * (1 + (c / np.sqrt(2)) ** 4)):
            return float(beta * c)


def _ite_gadget(qc: QuantumCircuit, anc, cbit, P: str, coeff: float) -> None:
    supp = [i for i, c in enumerate(P) if c != "I"]
    if not supp or np.isclose(coeff, 0.0):
        return
    mag = abs(coeff)
    for i in supp:
        if P[i] == "X":
            qc.h(i)
        elif P[i] == "Y":
            qc.sdg(i)
            qc.h(i)
    for a, b in zip(supp[:-1], supp[1:]):
        qc.cx(a, b)
    if coeff > 0:
        qc.x(supp[-1])
    qc.crx(2 * np.arccos(np.exp(-2 * mag)), supp[-1], anc)
    qc.measure(anc, cbit)
    qc.reset(anc)
    if coeff > 0:
        qc.x(supp[-1])
    for a, b in zip(supp[-2::-1], supp[-1:0:-1]):
        qc.cx(a, b)
    for i in reversed(supp):
        if P[i] == "X":
            qc.h(i)
        elif P[i] == "Y":
            qc.h(i)
            qc.s(i)


def _rte_gadget(qc: QuantumCircuit, P: str, theta: float) -> None:
    supp = [i for i, c in enumerate(P) if c != "I"]
    if not supp:
        return
    for i in supp:
        if P[i] == "X":
            qc.h(i)
        elif P[i] == "Y":
            qc.sdg(i)
            qc.h(i)
    for a, b in zip(supp[:-1], supp[1:]):
        qc.cx(a, b)
    qc.rz(2 * theta, supp[-1])
    for a, b in zip(supp[-2::-1], supp[-1:0:-1]):
        qc.cx(a, b)
    for i in reversed(supp):
        if P[i] == "X":
            qc.h(i)
        elif P[i] == "Y":
            qc.h(i)
            qc.s(i)


def _drop_ancilla(rho_dm: DensityMatrix | np.ndarray, anc_q: int) -> np.ndarray:
    mat = np.asarray(rho_dm.data) if isinstance(rho_dm, DensityMatrix) else np.asarray(rho_dm)
    n = int(np.log2(mat.shape[0]))
    axis = n - 1 - anc_q
    tens = mat.reshape([2] * n + [2] * n)
    tens = np.take(tens, 0, axis=axis)
    tens = np.take(tens, 0, axis=n + axis - 1)
    out = tens.reshape(2 ** (n - 1), 2 ** (n - 1))
    p = np.real(np.trace(out))
    return out / p if p > 0 else out


def _one_circuit_rho(
    n: int,
    psi0: np.ndarray,
    ite_sched: list[tuple[str, float]],
    rte_sched: list[tuple[str, float]],
    shots: int,
    noise_model: NoiseModel | None,
) -> tuple[np.ndarray | None, float]:
    q = QuantumRegister(n)
    anc = QuantumRegister(1)
    c = ClassicalRegister(len(ite_sched))
    qc = QuantumCircuit(q, anc, c)
    qc.initialize(psi0, q)
    for j, (P, coeff) in enumerate(ite_sched):
        _ite_gadget(qc, anc[0], c[j], P, coeff)
    for P, coeff in rte_sched:
        _rte_gadget(qc, P, coeff)
    qc.save_density_matrix(label="rho", pershot=True)
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    tqc = transpile(qc, sim)
    res = sim.run(tqc, shots=shots, memory=True).result()
    rhos = res.data(tqc)["rho"]
    mem = res.get_memory(tqc)
    keep = [k for k, bits in enumerate(mem) if set(bits) <= {"0"}]
    if not keep:
        return None, 0.0
    kept = [_drop_ancilla(rhos[k], n) for k in keep]
    return np.mean(kept, axis=0), len(keep) / shots


def _build_noise_model(p: float) -> NoiseModel | None:
    if p <= 0:
        return None
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["x", "h", "s", "sdg", "rz", "rx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx", "crx"])
    return nm


def _collect_metrics(rho: np.ndarray, ground_rho: np.ndarray, H: np.ndarray, spin_idx: int) -> tuple[float, float, float]:
    return trdist(rho, ground_rho), expect(rho, H), spin_cov_xx(rho, spin_idx)


def _sample_complexity_curve(
    trd_curve: np.ndarray, p_succ_curve: np.ndarray, beta: np.ndarray, epsilon: float, M_factor: int
) -> np.ndarray:
    out = np.full_like(beta, np.inf, dtype=float)
    mask = trd_curve <= epsilon
    out[mask] = M_factor / np.clip(p_succ_curve[mask], 1e-12, None)
    return out


def run_all_experiments(cfg: ExperimentConfig, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    betas = np.linspace(cfg.beta_min, cfg.beta_max, cfg.beta_num)
    terms = build_tfim_terms(cfg.n, cfg.J, cfg.g)
    H = build_hamiltonian_matrix(terms)
    evals, evecs = np.linalg.eigh(H)
    ground_rho = outer(evecs[:, 0])
    psi0 = np.ones(2**cfg.n, dtype=complex) / np.sqrt(2**cfg.n)
    exact = ExactSim(H)
    exact_trot = ExactSim(H, terms=terms, trotter_steps=cfg.trotter_steps, trotter_order=cfg.trotter_order)

    shape = (len(betas), 2)
    metrics_exact = {k: np.zeros(shape) for k in ("trd", "energy", "spincov")}
    metrics_circ_ideal = {k: np.zeros(shape) for k in ("trd", "energy", "spincov")}
    succ_ideal = np.zeros(shape)
    noisy = {
        p: {k: np.zeros(shape) for k in ("trd", "energy", "spincov")} for p in cfg.noise_levels
    }
    succ_noisy = {p: np.zeros(shape) for p in cfg.noise_levels}

    for i, beta in enumerate(betas):
        sampler = lambda b=beta: gcauchy_sample(b, rng)
        rho_e_ite = outer(exact.ite(psi0, beta))
        rho_e_rite = exact.rite(psi0, beta, cfg.M_exact, sampler)
        for j, rho in enumerate((rho_e_ite, rho_e_rite)):
            metrics_exact["trd"][i, j], metrics_exact["energy"][i, j], metrics_exact["spincov"][i, j] = (
                _collect_metrics(rho, ground_rho, H, cfg.spin_index)
            )

        ite_sched = trotter_schedule(terms, beta, cfg.trotter_steps, cfg.trotter_order)
        rho_i, p_i = _one_circuit_rho(cfg.n, psi0, ite_sched, [], cfg.shots, None)
        t = sampler()
        rte_sched = trotter_schedule(terms, t, cfg.trotter_steps, cfg.trotter_order)
        rho_r, p_r = _one_circuit_rho(cfg.n, psi0, ite_sched, rte_sched, cfg.shots, None)
        for j, (rho, ps) in enumerate(((rho_i, p_i), (rho_r, p_r))):
            succ_ideal[i, j] = ps
            if rho is None:
                metrics_circ_ideal["trd"][i, j] = np.nan
                metrics_circ_ideal["energy"][i, j] = np.nan
                metrics_circ_ideal["spincov"][i, j] = np.nan
            else:
                (
                    metrics_circ_ideal["trd"][i, j],
                    metrics_circ_ideal["energy"][i, j],
                    metrics_circ_ideal["spincov"][i, j],
                ) = _collect_metrics(rho, ground_rho, H, cfg.spin_index)

        for p in cfg.noise_levels:
            nm = _build_noise_model(p)
            rho_i_acc = []
            rho_r_acc = []
            ps_i, ps_r = [], []
            for _ in range(cfg.M_circuit):
                rho_i, pi = _one_circuit_rho(cfg.n, psi0, ite_sched, [], cfg.shots, nm)
                t = sampler()
                rte_sched = trotter_schedule(terms, t, cfg.trotter_steps, cfg.trotter_order)
                rho_r, pr = _one_circuit_rho(cfg.n, psi0, ite_sched, rte_sched, cfg.shots, nm)
                ps_i.append(pi)
                ps_r.append(pr)
                if rho_i is not None:
                    rho_i_acc.append(rho_i)
                if rho_r is not None:
                    rho_r_acc.append(rho_r)
            succ_noisy[p][i, 0] = float(np.mean(ps_i))
            succ_noisy[p][i, 1] = float(np.mean(ps_r))
            for j, acc in enumerate((rho_i_acc, rho_r_acc)):
                if not acc:
                    noisy[p]["trd"][i, j] = np.nan
                    noisy[p]["energy"][i, j] = np.nan
                    noisy[p]["spincov"][i, j] = np.nan
                else:
                    m = np.mean(acc, axis=0)
                    noisy[p]["trd"][i, j], noisy[p]["energy"][i, j], noisy[p]["spincov"][i, j] = (
                        _collect_metrics(m, ground_rho, H, cfg.spin_index)
                    )

    sample_cost = {
        "ideal": np.stack(
            [
                _sample_complexity_curve(
                    metrics_circ_ideal["trd"][:, j], succ_ideal[:, j], betas, cfg.target_trace_distance, 1 if j == 0 else cfg.M_circuit
                )
                for j in range(2)
            ],
            axis=1,
        )
    }
    for p in cfg.noise_levels:
        sample_cost[f"noise_{p:g}"] = np.stack(
            [
                _sample_complexity_curve(
                    noisy[p]["trd"][:, j], succ_noisy[p][:, j], betas, cfg.target_trace_distance, 1 if j == 0 else cfg.M_circuit
                )
                for j in range(2)
            ],
            axis=1,
        )

    payload = {
        "config": asdict(cfg),
        "betas": betas,
        "evals": evals,
        "exact": metrics_exact,
        "circuit_ideal": metrics_circ_ideal,
        "noisy": noisy,
        "success_ideal": succ_ideal,
        "success_noisy": succ_noisy,
        "sample_complexity": sample_cost,
    }
    np.savez_compressed(out_dir / "results.npz", payload=payload)
    return payload

