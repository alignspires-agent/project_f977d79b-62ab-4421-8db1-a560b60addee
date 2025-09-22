
import numpy as np
import sys
import logging
from typing import Tuple, List, Optional
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LPConformalPrediction:
    """
    Implementation of Conformal Prediction under Lévy–Prokhorov Distribution Shifts
    based on the paper methodology.
    """
    
    def __init__(self, epsilon: float = 0.1, rho: float = 0.05, alpha: float = 0.1):
        """
        Initialize the LP conformal prediction model.
        
        Parameters:
        epsilon (float): Local perturbation parameter (ε)
        rho (float): Global perturbation parameter (ρ)
        alpha (float): Significance level (1 - coverage)
        """
        self.epsilon = epsilon
        self.rho = rho
        self.alpha = alpha
        self.scores = None
        self.quantile = None
        
    def compute_scores(self, X_calib: np.ndarray, y_calib: np.ndarray) -> np.ndarray:
        """
        Compute conformity scores for calibration data.
        Using absolute error as a simple scoring function.
        
        Parameters:
        X_calib (np.ndarray): Calibration features
        y_calib (np.ndarray): Calibration targets
        
        Returns:
        np.ndarray: Conformity scores
        """
        try:
            # Simple mean prediction as placeholder - replace with actual model
            mean_pred = np.mean(y_calib)
            scores = np.abs(y_calib - mean_pred)
            return scores
        except Exception as e:
            logger.error(f"Error computing scores: {e}")
            sys.exit(1)
    
    def compute_worst_case_quantile(self, scores: np.ndarray) -> float:
        """
        Compute worst-case quantile under LP distribution shifts.
        
        Parameters:
        scores (np.ndarray): Conformity scores
        
        Returns:
        float: Worst-case quantile value
        """
        try:
            n = len(scores)
            sorted_scores = np.sort(scores)
            
            # Compute empirical quantile
            empirical_quantile_idx = int(np.ceil((1 - self.alpha) * (n + 1))) - 1
            empirical_quantile = sorted_scores[empirical_quantile_idx]
            
            # Apply LP robustness adjustments
            # Local perturbation: ε affects the quantile position
            local_shift = int(np.floor(self.epsilon * n))
            local_quantile_idx = min(empirical_quantile_idx + local_shift, n - 1)
            local_quantile = sorted_scores[local_quantile_idx]
            
            # Global perturbation: ρ affects the quantile value
            global_adjustment = self.rho * (np.max(scores) - np.min(scores))
            worst_case_quantile = local_quantile + global_adjustment
            
            return worst_case_quantile
        except Exception as e:
            logger.error(f"Error computing worst-case quantile: {e}")
            sys.exit(1)
    
    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray):
        """
        Fit the conformal prediction model on calibration data.
        
        Parameters:
        X_calib (np.ndarray): Calibration features
        y_calib (np.ndarray): Calibration targets
        """
        try:
            logger.info("Fitting LP conformal prediction model...")
            self.scores = self.compute_scores(X_calib, y_calib)
            self.quantile = self.compute_worst_case_quantile(self.scores)
            logger.info(f"Model fitted. Worst-case quantile: {self.quantile:.4f}")
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            sys.exit(1)
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for test data.
        
        Parameters:
        X_test (np.ndarray): Test features
        
        Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of prediction intervals
        """
        try:
            # Simple mean prediction as placeholder
            mean_pred = np.mean([np.mean(row) for row in X_test]) if X_test.ndim > 1 else np.mean(X_test)
            
            lower_bounds = mean_pred - self.quantile
            upper_bounds = mean_pred + self.quantile
            
            return lower_bounds, upper_bounds
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            sys.exit(1)
    
    def evaluate_coverage(self, y_true: np.ndarray, lower_bounds: np.ndarray, 
                         upper_bounds: np.ndarray) -> float:
        """
        Evaluate coverage of prediction intervals.
        
        Parameters:
        y_true (np.ndarray): True target values
        lower_bounds (np.ndarray): Lower bounds of intervals
        upper_bounds (np.ndarray): Upper bounds of intervals
        
        Returns:
        float: Coverage percentage
        """
        try:
            covered = np.sum((y_true >= lower_bounds) & (y_true <= upper_bounds))
            coverage = covered / len(y_true)
            return coverage
        except Exception as e:
            logger.error(f"Error evaluating coverage: {e}")
            sys.exit(1)

def generate_time_series_data(n_samples: int = 1000, trend_strength: float = 0.1, 
                             seasonality_period: int = 50, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data with different distribution characteristics.
    
    Parameters:
    n_samples (int): Number of samples to generate
    trend_strength (float): Strength of the linear trend
    seasonality_period (int): Period of seasonal component
    noise_level (float): Level of noise to add
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: Time indices and time series values
    """
    try:
        np.random.seed(42)
        time_indices = np.arange(n_samples)
        
        # Generate different components
        trend = trend_strength * time_indices
        seasonality = 2 * np.sin(2 * np.pi * time_indices / seasonality_period)
        noise = noise_level * np.random.randn(n_samples)
        
        # Combine components
        y = trend + seasonality + noise
        
        # Create feature matrix (time indices and lagged values)
        X = np.column_stack([time_indices, 
                            np.roll(y, 1),  # lag 1
                            np.roll(y, 2)]) # lag 2
        
        # Remove first two rows with NaN values
        X = X[2:]
        y = y[2:]
        time_indices = time_indices[2:]
        
        return X, y, time_indices
    except Exception as e:
        logger.error(f"Error generating time series data: {e}")
        sys.exit(1)

def generate_different_distributions(n_samples: int = 1000) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Generate datasets with different distribution characteristics.
    
    Parameters:
    n_samples (int): Number of samples per dataset
    
    Returns:
    List[Tuple[np.ndarray, np.ndarray, str]]: List of (X, y, distribution_type) tuples
    """
    datasets = []
    
    try:
        # 1. Normal distribution
        np.random.seed(42)
        X_normal = np.random.randn(n_samples, 5)
        y_normal = X_normal @ np.array([1.0, -0.5, 2.0, -1.0, 0.3]) + 0.1 * np.random.randn(n_samples)
        datasets.append((X_normal, y_normal, "Normal"))
        
        # 2. Exponential distribution
        X_exp = np.random.exponential(1.0, (n_samples, 3))
        y_exp = X_exp @ np.array([1.0, -2.0, 0.5]) + 0.1 * np.random.exponential(0.5, n_samples)
        datasets.append((X_exp, y_exp, "Exponential"))
        
        # 3. Uniform distribution
        X_unif = np.random.uniform(-2, 2, (n_samples, 4))
        y_unif = X_unif @ np.array([0.8, -1.2, 0.4, -0.6]) + 0.1 * np.random.uniform(-1, 1, n_samples)
        datasets.append((X_unif, y_unif, "Uniform"))
        
        # 4. Time series data
        X_ts, y_ts, _ = generate_time_series_data(n_samples)
        datasets.append((X_ts, y_ts, "TimeSeries"))
        
        # 5. Mixture distribution
        X_mix1 = np.random.normal(0, 1, (n_samples//2, 2))
        X_mix2 = np.random.normal(3, 0.5, (n_samples//2, 2))
        X_mix = np.vstack([X_mix1, X_mix2])
        y_mix = X_mix @ np.array([1.0, -0.8]) + 0.1 * np.random.randn(n_samples)
        datasets.append((X_mix, y_mix, "Mixture"))
        
        return datasets
        
    except Exception as e:
        logger.error(f"Error generating different distributions: {e}")
        sys.exit(1)

def simulate_distribution_shift(X: np.ndarray, y: np.ndarray, shift_type: str = "covariate") -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate different types of distribution shifts.
    
    Parameters:
    X (np.ndarray): Original features
    y (np.ndarray): Original targets
    shift_type (str): Type of shift ("covariate", "target", "both")
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: Shifted features and targets
    """
    try:
        if shift_type == "covariate":
            # Shift covariates only
            X_shifted = X + 0.3 * np.random.randn(*X.shape)
            y_shifted = y.copy()
        elif shift_type == "target":
            # Shift target distribution only
            X_shifted = X.copy()
            y_shifted = y + 0.4 * np.random.randn(len(y))
        else:  # both
            # Shift both covariates and target
            X_shifted = X + 0.3 * np.random.randn(*X.shape)
            y_shifted = y + 0.4 * np.random.randn(len(y))
        
        return X_shifted, y_shifted
    except Exception as e:
        logger.error(f"Error simulating distribution shift: {e}")
        sys.exit(1)

def run_experiment_on_dataset(X: np.ndarray, y: np.ndarray, dataset_name: str):
    """
    Run the LP conformal prediction experiment on a single dataset.
    
    Parameters:
    X (np.ndarray): Features
    y (np.ndarray): Targets
    dataset_name (str): Name of the dataset
    
    Returns:
    dict: Experiment results
    """
    try:
        # Split data into calibration and test sets
        split_idx = int(0.7 * len(X))
        X_calib, y_calib = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        # Initialize and fit the robust model
        model_robust = LPConformalPrediction(epsilon=0.1, rho=0.05, alpha=0.1)
        model_robust.fit(X_calib, y_calib)
        
        # Initialize and fit the non-robust model
        model_non_robust = LPConformalPrediction(epsilon=0.0, rho=0.0, alpha=0.1)
        model_non_robust.fit(X_calib, y_calib)
        
        # Test on original data
        lower_robust, upper_robust = model_robust.predict(X_test)
        coverage_robust_orig = model_robust.evaluate_coverage(y_test, lower_robust, upper_robust)
        
        lower_non_robust, upper_non_robust = model_non_robust.predict(X_test)
        coverage_non_robust_orig = model_non_robust.evaluate_coverage(y_test, lower_non_robust, upper_non_robust)
        
        # Test on shifted data (different shift types)
        results = {}
        for shift_type in ["covariate", "target", "both"]:
            X_test_shifted, y_test_shifted = simulate_distribution_shift(X_test, y_test, shift_type)
            
            # Robust model on shifted data
            coverage_robust_shifted = model_robust.evaluate_coverage(y_test_shifted, lower_robust, upper_robust)
            
            # Non-robust model on shifted data (need to re-predict with shifted features)
            lower_non_robust_shifted, upper_non_robust_shifted = model_non_robust.predict(X_test_shifted)
            coverage_non_robust_shifted = model_non_robust.evaluate_coverage(
                y_test_shifted, lower_non_robust_shifted, upper_non_robust_shifted)
            
            # Calculate coverage drops
            coverage_drop_robust = abs((1 - model_robust.alpha) - coverage_robust_shifted)
            coverage_drop_non_robust = abs((1 - model_non_robust.alpha) - coverage_non_robust_shifted)
            robustness_improvement = coverage_robust_shifted - coverage_non_robust_shifted
            
            results[shift_type] = {
                "coverage_robust_shifted": coverage_robust_shifted,
                "coverage_non_robust_shifted": coverage_non_robust_shifted,
                "coverage_drop_robust": coverage_drop_robust,
                "coverage_drop_non_robust": coverage_drop_non_robust,
                "robustness_improvement": robustness_improvement
            }
        
        return {
            "dataset": dataset_name,
            "coverage_robust_original": coverage_robust_orig,
            "coverage_non_robust_original": coverage_non_robust_orig,
            "shift_results": results,
            "worst_case_quantile_robust": model_robust.quantile,
            "worst_case_quantile_non_robust": model_non_robust.quantile
        }
        
    except Exception as e:
        logger.error(f"Error running experiment on {dataset_name}: {e}")
        sys.exit(1)

def main():
    """Main experiment function to test LP conformal prediction on multiple distributions."""
    logger.info("Starting LP Conformal Prediction Experiment on Multiple Distributions")
    
    # Generate datasets with different distributions
    logger.info("Generating datasets with different distributions...")
    datasets = generate_different_distributions(n_samples=1000)
    
    # Run experiments on each dataset
    all_results = []
    for i, (X, y, dist_type) in enumerate(datasets):
        logger.info(f"Testing distribution: {dist_type}")
        result = run_experiment_on_dataset(X, y, dist_type)
        all_results.append(result)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENT RESULTS")
    print("="*80)
    
    for result in all_results:
        print(f"\nDataset: {result['dataset']}")
        print(f"Robust Coverage (Original): {result['coverage_robust_original']:.4f}")
        print(f"Non-Robust Coverage (Original): {result['coverage_non_robust_original']:.4f}")
        print(f"Robust Worst-case Quantile: {result['worst_case_quantile_robust']:.4f}")
        print(f"Non-Robust Quantile: {result['worst_case_quantile_non_robust']:.4f}")
        
        for shift_type, shift_result in result['shift_results'].items():
            print(f"\n  Shift Type: {shift_type}")
            print(f"    Robust Coverage: {shift_result['coverage_robust_shifted']:.4f}")
            print(f"    Non-Robust Coverage: {shift_result['coverage_non_robust_shifted']:.4f}")
            print(f"    Robust Coverage Drop: {shift_result['coverage_drop_robust']:.4f}")
            print(f"    Non-Robust Coverage Drop: {shift_result['coverage_drop_non_robust']:.4f}")
            print(f"    Robustness Improvement: {shift_result['robustness_improvement']:.4f}")
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    robustness_improvements = []
    for result in all_results:
        for shift_result in result['shift_results'].values():
            robustness_improvements.append(shift_result['robustness_improvement'])
    
    avg_improvement = np.mean(robustness_improvements)
    std_improvement = np.std(robustness_improvements)
    min_improvement = np.min(robustness_improvements)
    max_improvement = np.max(robustness_improvements)
    
    print(f"Average Robustness Improvement: {avg_improvement:.4f}")
    print(f"Standard Deviation: {std_improvement:.4f}")
    print(f"Minimum Improvement: {min_improvement:.4f}")
    print(f"Maximum Improvement: {max_improvement:.4f}")
    
    # Final conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if avg_improvement > 0:
        print("✓ LP Robust Conformal Prediction demonstrates consistent robustness")
        print(f"✓ Average improvement over non-robust method: {avg_impro