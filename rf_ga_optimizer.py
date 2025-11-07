import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

try:
    from scipy.spatial import Delaunay
except ImportError as exc:  # pragma: no cover - defensive fallback
    raise SystemExit("scipy is required for convex hull computations") from exc


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


@dataclass
class SurrogateMetrics:
    rmse: float
    r2: float
    q90: float


class RandomForestSurrogate:
    """Random Forest surrogate with isotonic calibration."""

    def __init__(self, n_estimators: int = 600, max_depth: Optional[int] = None, random_state: int = RANDOM_STATE) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.metrics: Optional[SurrogateMetrics] = None

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> np.ndarray:
        deg_rad = np.deg2rad(df["Deg0"].to_numpy())
        sin_deg = np.sin(deg_rad)
        cos_deg = np.cos(deg_rad)
        features = np.column_stack(
            [
                df["L0"].to_numpy(),
                sin_deg,
                cos_deg,
                df["L1"].to_numpy(),
                df["L2"].to_numpy(),
            ]
        )
        return features

    def fit(self, df: pd.DataFrame) -> SurrogateMetrics:
        X = self._prepare_features(df)
        y = df["matched_L"].to_numpy()

        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_raw = np.zeros_like(y, dtype=float)
        for train_idx, valid_idx in kfold.split(X):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X[train_idx], y[train_idx])
            oof_raw[valid_idx] = model.predict(X[valid_idx])

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_raw, y)
        oof_calibrated = calibrator.transform(oof_raw)

        rmse = math.sqrt(mean_squared_error(y, oof_calibrated))
        r2 = r2_score(y, oof_calibrated)
        q90 = float(np.quantile(np.abs(oof_calibrated - y), 0.9))

        self.metrics = SurrogateMetrics(rmse=rmse, r2=r2, q90=q90)

        final_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        final_model.fit(X, y)

        self.model = final_model
        self.calibrator = calibrator

        print("[Surrogate] RMSE: {:.6f}".format(rmse))
        print("[Surrogate] R^2: {:.6f}".format(r2))
        print("[Surrogate] q90(|error|): {:.6f}".format(q90))

        return self.metrics

    def predict(self, L0: float, Deg0: float, L1: float, L2: float) -> Tuple[float, float, float]:
        if self.model is None or self.calibrator is None:
            raise RuntimeError("Model has not been fitted yet.")

        sin_deg = math.sin(math.radians(Deg0))
        cos_deg = math.cos(math.radians(Deg0))
        X = np.array([[L0, sin_deg, cos_deg, L1, L2]])
        raw_pred = self.model.predict(X)[0]

        # Tree-wise predictions for uncertainty
        tree_preds = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
        variance = float(np.var(tree_preds))
        uncertainty = math.sqrt(variance)

        calibrated_pred = float(self.calibrator.transform([raw_pred])[0])
        return calibrated_pred, uncertainty, variance


class TrustRegion:
    """Trust-region scoring using kNN distances and convex hull membership."""

    def __init__(self, L1_values: np.ndarray, L2_values: np.ndarray, n_neighbors: int = 5) -> None:
        points = np.column_stack([L1_values, L2_values])
        self.points = points
        self.n_neighbors = min(n_neighbors, len(points))
        if self.n_neighbors < 2:
            raise ValueError("Not enough points to build trust region.")

        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn.fit(points)

        distances, _ = self.nn.kneighbors(points)
        # Exclude distance to self (0.0)
        if distances.shape[1] > 1:
            avg_distances = distances[:, 1:].mean(axis=1)
        else:
            avg_distances = distances[:, 0]
        self.knn_threshold = float(np.quantile(avg_distances, 0.8))

        try:
            self.hull = Delaunay(points)
        except Exception:
            self.hull = None

    def inside_hull(self, point: np.ndarray) -> bool:
        if self.hull is None:
            return True
        return self.hull.find_simplex(point) >= 0

    def score(self, L1: float, L2: float) -> Tuple[float, float]:
        point = np.array([[L1, L2]])
        if not self.inside_hull(point):
            # Catastrophic penalty outside convex hull
            return 1e6, 1e6

        distances, _ = self.nn.kneighbors(point)
        if distances.shape[1] > 1:
            avg_dist = float(distances[:, 1:].mean())
        else:
            avg_dist = float(distances.mean())

        trust_score = avg_dist / (self.knn_threshold + 1e-9)
        penalty = 0.0
        if avg_dist > self.knn_threshold:
            overflow = (avg_dist - self.knn_threshold) / (self.knn_threshold + 1e-9)
            penalty += overflow * 5.0
        return trust_score, penalty


@dataclass
class Candidate:
    L1: float
    L2: float
    pred: float
    uncertainty: float
    variance: float
    trust: float
    penalty: float
    objective: float


class GeneticOptimizer:
    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], Candidate],
        bounds: np.ndarray,
        pop_size: int,
        n_generations: int,
        restarts: int,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.9,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.restarts = restarts
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.dim = bounds.shape[0]
        self.rng = np.random.default_rng(seed)

    def _initialize_population(self) -> np.ndarray:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        population = self.rng.uniform(lower, upper, size=(self.pop_size, self.dim))
        return population

    def _tournament(self, fitness: np.ndarray, k: int = 3) -> int:
        idx = self.rng.choice(len(fitness), size=k, replace=False)
        best_idx = idx[np.argmin(fitness[idx])]
        return best_idx

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return parent_a.copy(), parent_b.copy()
        alpha = self.rng.uniform(-0.1, 1.1, size=self.dim)
        child1 = alpha * parent_a + (1 - alpha) * parent_b
        child2 = alpha * parent_b + (1 - alpha) * parent_a
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.mutation_rate:
            span = self.bounds[:, 1] - self.bounds[:, 0]
            noise = self.rng.normal(0.0, 0.05, size=self.dim) * span
            individual = individual + noise
        return np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])

    def run(self) -> List[Candidate]:
        best_candidates: List[Candidate] = []
        for restart in range(self.restarts):
            population = self._initialize_population()
            candidates = [self.objective_fn(ind) for ind in population]
            fitness = np.array([cand.objective for cand in candidates])

            for _ in range(self.n_generations):
                new_population = []
                while len(new_population) < self.pop_size:
                    parent1 = population[self._tournament(fitness)]
                    parent2 = population[self._tournament(fitness)]
                    child1, child2 = self._crossover(parent1, parent2)
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    new_population.append(child1)
                    if len(new_population) < self.pop_size:
                        new_population.append(child2)
                population = np.array(new_population)
                candidates = [self.objective_fn(ind) for ind in population]
                fitness = np.array([cand.objective for cand in candidates])

            best_idx = int(np.argmin(fitness))
            best_candidates.append(candidates[best_idx])
            # Re-seed RNG to diversify restarts
            self.rng = np.random.default_rng(self.rng.integers(0, 1_000_000))

        return best_candidates


def augment_data(df: pd.DataFrame, copies: int = 3, l0_frac_noise: float = 0.01, deg_noise: float = 0.5) -> pd.DataFrame:
    augmented_rows = [df]
    l0_range = df["L0"].max() - df["L0"].min()
    l0_scale = max(l0_range * l0_frac_noise, 1e-6)
    for i in range(copies):
        noisy = df.copy()
        noisy["L0"] += np.random.normal(0.0, l0_scale, size=len(df))
        noisy["Deg0"] += np.random.normal(0.0, deg_noise, size=len(df))
        augmented_rows.append(noisy)
    augmented = pd.concat(augmented_rows, ignore_index=True)
    return augmented


def safe_threshold(q90: float) -> float:
    return max(0.0, 0.06 - q90)


def load_dataset(data_path: Path, augmented_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    augmented = augment_data(df)
    augmented.to_csv(augmented_path, index=False)
    print(f"[Data] Augmented data saved to {augmented_path}")
    return augmented


def build_bounds(df: pd.DataFrame) -> np.ndarray:
    bounds = []
    for col in ["L1", "L2"]:
        values = df[col].to_numpy()
        col_min = values.min()
        col_max = values.max()
        span = col_max - col_min
        buffer = span * 0.02
        bounds.append((col_min - buffer, col_max + buffer))
    return np.array(bounds)


def find_diverse_solutions(
    surrogate: RandomForestSurrogate,
    trust_region: TrustRegion,
    bounds: np.ndarray,
    L0: float,
    Deg0: float,
    ga_pop: int,
    ga_gens: int,
    restarts: int,
    safe_thr: float,
    epsilon: float = 0.03,
) -> List[Candidate]:
    def objective(individual: np.ndarray) -> Candidate:
        L1, L2 = float(individual[0]), float(individual[1])
        pred, uncertainty, variance = surrogate.predict(L0, Deg0, L1, L2)
        trust_score, penalty = trust_region.score(L1, L2)
        if penalty >= 1e6 or trust_score >= 1e6:
            objective_value = pred + penalty
        else:
            objective_value = pred + 0.3 * uncertainty + 0.4 * trust_score + penalty
        return Candidate(
            L1=L1,
            L2=L2,
            pred=pred,
            uncertainty=uncertainty,
            variance=variance,
            trust=trust_score,
            penalty=penalty,
            objective=objective_value,
        )

    optimizer = GeneticOptimizer(
        objective_fn=objective,
        bounds=bounds,
        pop_size=ga_pop,
        n_generations=ga_gens,
        restarts=restarts,
    )

    raw_candidates = optimizer.run()

    feasible = [cand for cand in raw_candidates if cand.pred <= safe_thr and cand.penalty < 1e6]
    feasible.sort(key=lambda c: (c.pred, c.objective))

    diverse: List[Candidate] = []
    for cand in feasible:
        if all(math.hypot(cand.L1 - kept.L1, cand.L2 - kept.L2) >= epsilon for kept in diverse):
            diverse.append(cand)
        if len(diverse) >= 3:
            break
    return diverse


FAST_MODE = True
DATA_FILE = Path(__file__).resolve().parent / "200小样本.csv"
AUGMENTED_FILE = Path(__file__).resolve().parent / "200小样本_增强.csv"

if FAST_MODE:
    GA_POP = 20
    GA_GENS = 25
    RESTARTS = 3
    N_DEMO = 5
else:
    GA_POP = 40
    GA_GENS = 60
    RESTARTS = 8
    N_DEMO = 20


def main() -> None:
    augmented_df = load_dataset(DATA_FILE, AUGMENTED_FILE)

    surrogate = RandomForestSurrogate()
    metrics = surrogate.fit(augmented_df)
    safe_thr = safe_threshold(metrics.q90)
    print(f"[Surrogate] Safe threshold: {safe_thr:.6f}")

    trust_region = TrustRegion(augmented_df["L1"].to_numpy(), augmented_df["L2"].to_numpy())
    bounds = build_bounds(augmented_df)

    print("[GA] Bounds for L1/L2:", bounds)
    print(f"[GA] GA_POP={GA_POP}, GA_GENS={GA_GENS}, RESTARTS={RESTARTS}")

    original_df = pd.read_csv(DATA_FILE)
    demo_rows = min(N_DEMO, len(original_df))
    for idx in range(demo_rows):
        row = original_df.iloc[idx]
        print("\n[Demo] Sample #{} (L0={:.4f}, Deg0={:.4f})".format(idx, row["L0"], row["Deg0"]))
        solutions = find_diverse_solutions(
            surrogate=surrogate,
            trust_region=trust_region,
            bounds=bounds,
            L0=row["L0"],
            Deg0=row["Deg0"],
            ga_pop=GA_POP,
            ga_gens=GA_GENS,
            restarts=RESTARTS,
            safe_thr=safe_thr,
        )
        if not solutions:
            print("  No feasible solutions under safe threshold.")
            continue
        for rank, sol in enumerate(solutions, start=1):
            print(
                "  Solution #{rank}: L1={L1:.6f}, L2={L2:.6f}, pred={pred:.6f}, unc={unc:.6f}, trust={trust:.6f}, penalty={penalty:.6f}, objective={objective:.6f}".format(
                    rank=rank,
                    L1=sol.L1,
                    L2=sol.L2,
                    pred=sol.pred,
                    unc=sol.uncertainty,
                    trust=sol.trust,
                    penalty=sol.penalty,
                    objective=sol.objective,
                )
            )


if __name__ == "__main__":
    main()
