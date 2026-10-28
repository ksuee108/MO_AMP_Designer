import os
import pandas as pd
import numpy as np

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2

from sklearn.preprocessing import MinMaxScaler
from pymoo.visualization.scatter import Scatter
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair

from BioAnalysis import Bio_analysis
import matplotlib.pyplot as plt
import streamlit as st
from math import comb
from functools import lru_cache

AA_STR = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_INDEX = {a: i for i, a in enumerate(AA_STR)}


# =========================================================
# Initial population seeded from the dataset
# =========================================================

def clean_seed_sequences(seqs):
    """Keep unique peptides that contain only the 20 standard amino acids."""
    if seqs is None:
        return []
    out = []
    for s in seqs:
        if isinstance(s, str):
            s = s.strip().upper()
            if s and set(s) <= set(AA_STR):
                out.append(s)
    return list(dict.fromkeys(out))


def encode_to_length(seq: str, length: int):
    """Fit one dataset peptide to the design length.

    longer  -> random contiguous window of `length` residues
    shorter -> the peptide followed by random residues
    """
    if len(seq) > length:
        start = np.random.randint(0, len(seq) - length + 1)
        seq = seq[start:start + length]
    x = [AA_INDEX[a] for a in seq]
    if len(x) < length:
        x += list(np.random.randint(0, len(AA_STR), size=length - len(x)))
    return x


class DatasetSampling(Sampling):
    """Initial population = `fraction` dataset peptides + random peptides (rest).

    Uses numpy's global RNG, which pymoo seeds from minimize(seed=...), so runs are reproducible.
    """

    def __init__(self, seed_sequences, fraction=0.5):
        super().__init__()
        self.seed_sequences = clean_seed_sequences(seed_sequences)
        self.fraction = float(np.clip(fraction, 0.0, 1.0))

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.randint(0, len(AA_STR), size=(n_samples, problem.n_var))
        k = int(round(self.fraction * n_samples)) if self.seed_sequences else 0
        if k > 0:
            pick = np.random.choice(len(self.seed_sequences), size=k,
                                    replace=len(self.seed_sequences) < k)
            for row, idx in enumerate(pick):
                X[row] = encode_to_length(self.seed_sequences[idx], problem.n_var)
        return X


def int_ops(sampling=None):
    return dict(
        sampling=sampling if sampling is not None else IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=3.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )


def auto_partitions(n_obj: int, max_points: int) -> int:
    # largest number of Das-Dennis partitions whose reference-point count does not exceed max_points
    p = 1
    while comb(p + n_obj, n_obj - 1) <= max_points:
        p += 1
    return p


def is_maximize(name: str, mode: str) -> bool:
    """Gravy: hydrophobicity = Maximize GRAVY；Other: Maximize = Maximize"""
    if name == "Gravy":
        return mode.lower() == "hydrophobicity"
    return mode.lower() == "maximize"


def decode_sequence(x) -> str:
    return ''.join(AA_STR[min(max(int(round(v)), 0), len(AA_STR) - 1)] for v in x)


@lru_cache(maxsize=50000)
def _props_cached(peptide: str) -> tuple:
    return tuple(_compute_properties_raw(peptide).items())


def compute_properties(peptide: str) -> dict:
    return dict(_props_cached(peptide))


def _compute_properties_raw(peptide: str) -> dict:
    """compute all properties at once (secondary structure is computed only once)"""
    ba = Bio_analysis(peptide)
    helix, turn, sheet = ba.get_secondary_structure_fraction()
    return {
        'Gravy': ba.get_gravy(),
        'Instability Index': ba.get_instability_index(),
        'Aliphatic Index': ba.get_aliphatic_index(),
        'Isoelectric point': ba.get_isoelectric_point(),
        'Net charge': ba.get_net_charge(),
        'Molecular Weight': ba.get_molecular_weight(),
        'Charge at pH': ba.get_charge_at_pH(),
        'Aromaticity': ba.get_aromaticity(),
        'Secondary structure fraction Helix': helix,
        'Secondary structure fraction Turn': turn,
        'Secondary structure fraction Sheet': sheet,
        'Boman Index': ba.get_boman_index(),
        'Amphipathicity': ba.get_amphipathicity(),
        'Correlation': ba.get_auto_correlation(),
        'Covariance': ba.get_auto_covariance(),
        'Hydrophobic Moenet': ba.get_hydrophobic_moenet(),
        'Mass': ba.get_mass(),
        'Mz': ba.get_mz(),
        'Sequence Length': ba.get_sequenceLength(),
    }


def build_ref_points(opt, directions, targets: dict) -> np.ndarray:
    """targets: {'Net charge': 5, 'Instability Index': 20}, return shape (1, n_obj)"""
    pt = []
    for name in opt:
        v = float(targets[name])
        pt.append(-v if is_maximize(name, directions[name]) else v)
    return np.array([pt])


class MyProblemWithData(Problem):
    def __init__(self, sequences, optimization_directions=None, sequence_length=12,
                 opt=2, constraint_dict_list=None, *args, **kwargs):
        n_constr = len(constraint_dict_list) if constraint_dict_list else 0
        super().__init__(n_var=sequence_length, n_obj=len(opt),
                         n_constr=n_constr, xl=0, xu=19, vtype=int)
        self.sequences = sequences
        self.optimization_directions = optimization_directions
        self.sequence_length = sequence_length
        self.opt = opt
        self.constraint_dict_list = constraint_dict_list or []
        self.evaluated = {}

    def _evaluate(self, X, out, *args, **kwargs):
        n_samples = X.shape[0]
        F = np.zeros((n_samples, len(self.opt)))
        G = np.zeros((n_samples, len(self.constraint_dict_list)))

        for i in range(n_samples):
            seq = decode_sequence(X[i])
            props = compute_properties(seq)
            self.evaluated.setdefault(seq, props)

            for k, name in enumerate(self.opt):
                v = props[name]
                F[i, k] = -v if is_maximize(name, self.optimization_directions[name]) else v

            for j, c in enumerate(self.constraint_dict_list):
                v = float(props[c["Feature"]])
                limit = float(c["Value"])
                G[i, j] = (v - limit) if c["Type"] == "max" else (limit - v)

        out["F"] = F
        if self.constraint_dict_list:
            out["G"] = G


def amino_acid_percentage(algo: str, df: pd.DataFrame):
    """
    Visualize global amino acid composition of Pareto-optimal AMP sequences
    generated by a specific multi-objective optimization algorithm.
    """

    if df is None or df.empty:
        st.warning(f"No optimized sequences available for {algo}")
        return

    if "Sequence" not in df.columns:
        st.error(f"'Sequence' column not found in optimized results for {algo}")
        return

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    amino_acid_counts = {aa: 0 for aa in amino_acids}
    total_amino_acids = 0

    for seq in df["Sequence"]:
        for aa in seq:
            if aa in amino_acid_counts:
                amino_acid_counts[aa] += 1
                total_amino_acids += 1

    if total_amino_acids == 0:
        st.warning(f"No valid amino acids found in sequences for {algo}")
        return

    percentages = {
        aa: (count / total_amino_acids) * 100
        for aa, count in amino_acid_counts.items()
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(percentages.keys(), percentages.values())
    ax.set_title(f"Amino Acid Composition of Pareto-optimal Sequences ({algo})")
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, max(percentages.values()) * 1.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def pick_best_index(df, optimization_directions):
    objectives = list(optimization_directions.keys())
    X = df[objectives].values.astype(float)
    for k, name in enumerate(objectives):
        if is_maximize(name, optimization_directions[name]):
            X[:, k] = -X[:, k]
    Xn = MinMaxScaler().fit_transform(X)
    return int(np.argmin(np.linalg.norm(Xn, axis=1)))


def plot_scatter_matrix(algo, df, optimization_directions):
    objectives = list(optimization_directions.keys())
    missing = [o for o in objectives if o not in df.columns]
    if missing:
        st.error(f"{algo} missing objectives: {missing}")
        return

    X = df[objectives].values
    n = len(objectives)
    best = pick_best_index(df, optimization_directions)

    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.2 * n), squeeze=False)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, objectives[i], ha="center", va="center",
                        fontsize=11, wrap=True)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.scatter(X[:, j], X[:, i], s=12, color="tab:blue")
                ax.scatter(X[best, j], X[best, i], s=40, color="red", zorder=3)
                ax.set_xlabel(objectives[j], fontsize=8)
                ax.set_ylabel(objectives[i], fontsize=8)
                ax.tick_params(labelsize=7)

    fig.suptitle(f"Pareto-optimal Solutions - {algo}", fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_pareto_fronts_many(algo: str, df: pd.DataFrame, optimization_directions: dict):
    """
    Visualize Pareto-optimal solutions using parallel coordinates and scatter matrix.
    """

    if df is None or df.empty:
        st.warning(f"No Pareto results available for {algo}")
        return

    selected_cols = list(optimization_directions.keys())

    missing_cols = [c for c in selected_cols if c not in df.columns]
    if missing_cols:
        st.error(f"{algo} missing objective columns: {missing_cols}")
        return

    plot_df = df[selected_cols].copy()

    X = plot_df.values
    n_samples, n_objectives = X.shape

    # ---------- Parallel Coordinates ----------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    st.markdown(f"### 🔹 Parallel Coordinates Plot ({algo})")
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(n_samples):
        ax.plot(range(n_objectives), X_scaled[i],
                color="tab:blue", alpha=0.4, linewidth=1)

    ax.set_xticks(range(n_objectives))
    ax.set_xticklabels(selected_cols, rotation=45, ha="right")
    ax.set_ylabel("Normalized Objective Value")
    ax.set_title(f"Pareto-optimal Solutions – {algo}")
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # ---------- Scatter Matrix ----------
    st.markdown(f"### 🔹 Scatter Matrix Plot ({algo})")
    plot_scatter_matrix(algo, plot_df, optimization_directions)


def plot_pareto_fronts_multi(algo: str, df: pd.DataFrame, optimization_directions: dict):
    """
    Visualize Pareto fronts for 2D / 3D objectives.
    """

    if df is None or df.empty:
        st.warning(f"No optimization results for {algo}")
        return

    objectives = list(optimization_directions.keys())
    missing = [o for o in objectives if o not in df.columns]

    if missing:
        st.error(f"{algo} missing objectives: {missing}")
        return

    plot_df = df[objectives].copy()

    X = plot_df.values
    plot = Scatter(tight_layout=True, labels=objectives)
    plot.add(X, s=30, color="red")
    plot.do()
    st.pyplot(plot.fig)
    plt.close(plot.fig)

    if len(objectives) == 3:
        plot_scatter_matrix(algo, plot_df, optimization_directions)


# ----------------------------
class algorithms_setup():
    def __init__(self, df, algorithms_list, pop_size, generations,
                 optimization_directions, length, opt, constraint_dict_list,
                 ref_targets=None, seed=42, init_fraction=0.5):
        self.sequences = df['Sequence'] if df is not None and 'Sequence' in df.columns else None
        self.algorithms_list = list(algorithms_list)
        self.pop_size = pop_size
        self.generations = generations
        self.optimization_directions = optimization_directions
        self.length = length
        self.opt = opt
        self.constraint_dict_list = constraint_dict_list
        self.ref_targets = ref_targets
        self.seed = seed
        self.init_fraction = init_fraction

    def _new_problem(self):
        return MyProblemWithData(
            sequences=self.sequences,
            optimization_directions=self.optimization_directions,
            sequence_length=self.length,
            opt=self.opt,
            constraint_dict_list=self.constraint_dict_list,
        )

    def run_optimization(self):
        st.write("Maximized objectives will be multiplied by -1 to convert them into "
                 "minimization problems for optimization, and the results will be "
                 "restored when displayed.")

        # Initial population: a fraction of it comes from the selected / uploaded dataset
        seed_seqs = clean_seed_sequences(self.sequences)
        if seed_seqs and self.init_fraction > 0:
            sampling = DatasetSampling(seed_seqs, self.init_fraction)
            n_seeded = int(round(self.init_fraction * self.pop_size))
            st.info(f"Initial population: ~{n_seeded} of {self.pop_size} peptides are drawn from "
                    f"{len(seed_seqs)} dataset sequences (fitted to length {self.length}); "
                    f"the rest are random.")
        else:
            sampling = IntegerRandomSampling()
            st.info("Initial population: fully random (no dataset sequences used).")

        # Das-Dennis reference directions (NSGA-III / U-NSGA-III)
        n_partitions = auto_partitions(len(self.opt), self.pop_size)
        ref_dirs = get_reference_directions(
            "das-dennis", n_dim=len(self.opt), n_partitions=n_partitions
        )

        # User-defined reference point (R-NSGA-II / R-NSGA-III)
        ref_pts = None
        wants_r = [a for a in ("R-NSGA-II", "R-NSGA-III") if a in self.algorithms_list]
        if wants_r:
            if self.ref_targets and all(n in self.ref_targets for n in self.opt):
                ref_pts = build_ref_points(
                    self.opt, self.optimization_directions, self.ref_targets
                )
            else:
                st.warning("R-NSGA-II / R-NSGA-III need a target value for every "
                           "objective. They are skipped.")
                self.algorithms_list = [a for a in self.algorithms_list if a not in wants_r]

        # Build lazily so that only the selected algorithms are instantiated
        factories = {
            'NSGA-II':     lambda: NSGA2(pop_size=self.pop_size, **int_ops(sampling)),
            'NSGA-III':    lambda: NSGA3(ref_dirs=ref_dirs, pop_size=self.pop_size, **int_ops(sampling)),
            'U-NSGA-III':  lambda: UNSGA3(ref_dirs=ref_dirs, pop_size=self.pop_size, **int_ops(sampling)),
            'R-NSGA-II':   lambda: RNSGA2(ref_points=ref_pts, pop_size=self.pop_size, **int_ops(sampling)),
            'R-NSGA-III':  lambda: RNSGA3(ref_points=ref_pts,
                                          pop_per_ref_point=max(1, self.pop_size // len(ref_pts)),
                                          **int_ops(sampling)),
            'AGE-MOEA':    lambda: AGEMOEA(pop_size=self.pop_size, **int_ops(sampling)),
            'AGE-MOEA-II': lambda: AGEMOEA2(pop_size=self.pop_size, **int_ops(sampling)),
        }
        self.algorithms = [
            (make(), name) for name, make in factories.items()
            if name in self.algorithms_list
        ]

    def run(self):
        st.info("Starting optimization...")
        st.session_state.optimization_results = {}

        for algorithm, name in self.algorithms:
            problem = self._new_problem()          # fresh evaluation log for every algorithm
            kwargs = dict(termination=('n_gen', self.generations),
                          verbose=False, save_history=False)
            res = minimize(problem, algorithm, seed=self.seed, **kwargs)
            all_df = pd.DataFrame([{'Sequence': s, **p, 'model': name}
                                   for s, p in problem.evaluated.items()])

            if res.X is None:
                st.warning(f"{name}: Can't find feasible solutions that satisfy the constraints. "
                           f"Please relax the constraints.")
                continue

            X_opt = np.atleast_2d(res.X)
            sequences = [decode_sequence(x) for x in X_opt]

            rows = []
            for s in sequences:
                props = compute_properties(s)
                props['Sequence'] = s
                props['model'] = name
                rows.append(props)

            pareto_df = pd.DataFrame(rows)
            pareto_df = pareto_df.drop_duplicates(subset='Sequence').reset_index(drop=True)

            res_dict = pareto_df[list(self.optimization_directions.keys())].copy()

            merged_df = pareto_df.copy()
            cols = ['Sequence'] + [c for c in merged_df.columns if c not in ('Sequence', 'model')] + ['model']
            merged_df = merged_df[cols]
            all_df = all_df[cols]

            st.session_state.optimization_results[name] = {
                "res_dict": res_dict,
                "pareto_df": all_df,        # all distinct peptides evaluated in this run
                "merged_df": merged_df,     # final Pareto-optimal solutions
                "directions": dict(self.optimization_directions),
            }
        return list(st.session_state.optimization_results.keys())