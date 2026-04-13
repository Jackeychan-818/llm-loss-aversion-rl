"""
Loss Aversion Model Estimation Module

This module implements NLS, SMS (Bootstrap), and MLE estimation methods
for analyzing loss aversion behavior in LLM choice experiments.

Models:
- Model A: Nonlinear Least Squares (NLS)
- Model B: Smoothed Maximum Score with Bootstrap standard errors
- Model C: Maximum Likelihood Estimation (Logit)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from itertools import product
from scipy.optimize import curve_fit, minimize
from scipy.stats import gaussian_kde, logistic, chi2
from scipy.optimize import approx_fprime
from statsmodels.tools.numdiff import approx_hess
from linearmodels.panel import PooledOLS
from joblib import Parallel, delayed


class LossAversionModel:
    """
    Loss Aversion Model for analyzing LLM choice behavior.
    
    Estimates parameters:
    - lambda: loss aversion coefficient
    - eta: status quo bias (intercept)
    - alpha: item-specific utility parameters
    - beta: attribute-specific utility parameters
    """

    def __init__(self, Model_name, feature, robust_model, input_X, input_Y, T, 
                 starting='ols', is_two=False):
        """
        Initialize the Loss Aversion Model.
        
        Parameters:
        -----------
        Model_name : str
            Name of the model (e.g., 'GPT-4', 'Llama-70B')
        feature : str
            Feature type ('baseline', 'debias', 'forced')
        robust_model : int
            Model specification (1, 2, or 3)
        input_X : str
            Filename for X perspective data
        input_Y : str
            Filename for Y perspective data
        T : float
            Temperature parameter (0 for deterministic, >0 for probabilistic)
        starting : str
            Initial parameter method ('ols', 'zero', 'prior', 'nls')
        is_two : bool
            Whether this is a two-stage estimation
        """
        self.Model_name = Model_name
        self.feature = feature
        self.robust_model = robust_model
        self.input_X = input_X
        self.input_Y = input_Y
        self.T = T
        self.starting = starting
        self.is_two = is_two
        
        # Load data
        base_path = f'{feature}/{Model_name}'
        with open(f'{base_path}/{input_X}', 'r', encoding='utf-8') as f:
            self.output_data_X = json.load(f)
        with open(f'{base_path}/{input_Y}', 'r', encoding='utf-8') as f:
            self.output_data_Y = json.load(f)
        
        # Create output directory
        os.makedirs(self._get_output_dir(), exist_ok=True)
        
        # Process and prepare data
        self._prepare_data()

    # =========================================================================
    # Helper Methods for Path and Parameter Construction
    # =========================================================================
    
    def _get_output_dir(self):
        """Get the output directory path based on configuration."""
        if self.is_two:
            return f'{self.feature}/{self.Model_name}/is_two/Model_{self.robust_model}'
        return f'{self.feature}/{self.Model_name}/Model_{self.robust_model}'
    
    def _get_output_path(self, filename):
        """Get full output file path."""
        return os.path.join(self._get_output_dir(), filename)
    
    def _get_param_names(self):
        """Get parameter names list based on model specification."""
        base_names = ['lambda', 'eta']
        alpha_names = [f'alpha_{i}' for i in range(2, self.num_items + 1)]
        beta_names = ['beta_1,2', 'beta_1,3', 'beta_2,1', 'beta_2,2', 
                      'beta_2,3', 'beta_3,1', 'beta_3,2', 'beta_3,3']
        
        if self.robust_model == 3:
            return base_names + alpha_names + beta_names + ['gamma']
        return base_names + alpha_names + beta_names
    
    def _get_model_filename(self, which_model):
        """Get the results filename for a specific model."""
        filenames = {
            'A': f'{self.Model_name}_NLS_estimation_T1(Model A).csv',
            'B': f'{self.Model_name}_sms_results_bootstrap(Model B).csv',
            'C': f'{self.Model_name}_Logit_MLE_Results_Final(Model C).csv'
        }
        return self._get_output_path(filenames.get(which_model, ''))

    # =========================================================================
    # Data Preparation
    # =========================================================================
    
    def _parse_yes_no_prob(self, item):
        """Parse Yes/No probability from item output if not already present."""
        if 'Yes / No prob' in item:
            return item['Yes / No prob']
        
        output = item.get('output', '')
        try:
            # Try "Yes/No" format first
            response = output.split("Yes/No")[1].lower()
        except:
            # Fall back to "Yes / No" format
            response = output.split("Yes / No")[1].lower()
        
        return (0.75, 0.25) if "yes" in response else (0.25, 0.75)
    
    def _prepare_data(self):
        """Prepare data matrices for estimation."""
        # Detect number of items from data
        all_item_nums = set()
        for item in self.output_data_X:
            all_item_nums.add(item['X_num'])
            all_item_nums.add(item['Y_num'])
        num_items_detected = max(all_item_nums) + 1  # 0-indexed
        
        # Count entries to process (avoid double-counting pairs)
        entries_to_process = sum(1 for item in self.output_data_X 
                                  if item['X_num'] < item['Y_num'])
        num_samples = entries_to_process * 2
        
        # Initialize arrays
        num_features = num_items_detected + 8
        X = np.zeros([num_samples, num_features])
        y = np.random.rand(num_samples)
        X_item_id, Y_item_id = [], []
        X_attr_ij, Y_attr_kl = [], []
        
        # Process each pair
        samples = 0
        for idx in range(len(self.output_data_X)):
            X_item = self.output_data_X[idx]
            Y_item = self.output_data_Y[idx]
            
            # Parse Yes/No probabilities if needed
            if 'Yes / No prob' not in X_item:
                self.output_data_X[idx]['Yes / No prob'] = self._parse_yes_no_prob(X_item)
            if 'Yes / No prob' not in Y_item:
                self.output_data_Y[idx]['Yes / No prob'] = self._parse_yes_no_prob(Y_item)
            
            X_item = self.output_data_X[idx]
            Y_item = self.output_data_Y[idx]
            i, j, k, l = X_item['attr']
            
            if X_item['X_num'] < X_item['Y_num']:
                # Process X perspective
                yes, no = X_item['Yes / No prob']
                if self.T == 0:
                    yes, no = (1, 0) if yes > no else (0, 1)
                
                X[samples][X_item['X_num']] = 1
                X[samples][X_item['Y_num']] = -1
                if i * 3 + j > 0:
                    X[samples][num_items_detected - 1 + i * 3 + j] += 1
                if k * 3 + l > 0:
                    X[samples][num_items_detected - 1 + k * 3 + l] -= 1
                y[samples] = no
                X_item_id.append(X_item['X_num'] + 1)
                Y_item_id.append(X_item['Y_num'] + 1)
                X_attr_ij.append(f'({i+1}, {j+1})')
                Y_attr_kl.append(f'({k+1}, {l+1})')
                samples += 1
                
                # Process Y perspective
                yes, no = Y_item['Yes / No prob']
                if self.T == 0:
                    yes, no = (1, 0) if yes > no else (0, 1)
                
                X[samples][Y_item['X_num']] = -1
                X[samples][Y_item['Y_num']] = 1
                if i * 3 + j > 0:
                    X[samples][num_items_detected - 1 + i * 3 + j] -= 1
                if k * 3 + l > 0:
                    X[samples][num_items_detected - 1 + k * 3 + l] += 1
                y[samples] = no
                X_item_id.append(X_item['Y_num'] + 1)
                Y_item_id.append(X_item['X_num'] + 1)
                X_attr_ij.append(f'({k+1}, {l+1})')
                Y_attr_kl.append(f'({i+1}, {j+1})')
                samples += 1
        
        self.num_items = num_items_detected
        self.num_attr_combos = 9  # 3x3 = 9
        
        # Number of observation cases = samples / 2 (each case has X and Y perspective)
        num_obs_cases = samples // 2
        
        # Create DataFrame with multi-index
        # Each "pair" in the panel is one observation case (item pair + attr combo)
        # comb_id 1 = X perspective, comb_id 2 = Y perspective
        print(f"Observations: {len(X_item_id)}, Cases: {num_obs_cases}")
        data = pd.DataFrame({
            'pair_id': np.repeat([f'P{i:05d}' for i in range(1, num_obs_cases + 1)], 2),
            'comb_id': np.tile(np.arange(1, 3), num_obs_cases),
            'y': y[:samples],
            'X_item_id': np.array(X_item_id),
            'Y_item_id': np.array(Y_item_id),
            'X_attr_ij': np.array(X_attr_ij),
            'Y_attr_kl': np.array(Y_attr_kl)
        })
        data = data.set_index(['pair_id', 'comb_id'])
        
        # Create attribute difference variables
        attr_codes = [str((i, j)) for i in range(1, 4) for j in range(1, 4)]
        attr_diff_vars = [f'Attr_Diff_{c}' for c in attr_codes if c != '(1, 1)']
        new_cols = {
            f'Attr_Diff_{attr_code}': ((data['X_attr_ij'] == attr_code).astype(int) -
                                       (data['Y_attr_kl'] == attr_code).astype(int))
            for attr_code in attr_codes if attr_code != '(1, 1)'
        }

        # Create item difference variables
        item_diff_vars = [f'Item_Diff_{item_id}' for item_id in range(2, self.num_items + 1)]
        for item_id in range(2, self.num_items + 1):
            new_cols[f'Item_Diff_{item_id}'] = ((data['X_item_id'] == item_id).astype(int) -
                                                 (data['Y_item_id'] == item_id).astype(int))
        new_cols['Item_Diff_1'] = ((data['X_item_id'] == 1).astype(int) -
                                    (data['Y_item_id'] == 1).astype(int))
        new_cols['const'] = 1
        data = pd.concat([data, pd.DataFrame(new_cols, index=data.index)], axis=1)
        
        # Store cleaned data
        exog_vars = item_diff_vars + ['Item_Diff_1'] + attr_diff_vars
        self.data_clean = data[exog_vars + ['y']].dropna()
        self.dependent = self.data_clean['y']
        self.exog_final = sm.add_constant(self.data_clean[item_diff_vars + attr_diff_vars])
        self.N = len(self.dependent)
        
        # Create indicator matrix W for nonlinear estimation
        item_ids = np.arange(1, self.num_items + 1)
        attr_codes_tuples = [(i, j) for i in range(1, 4) for j in range(1, 4)]
        W_dict = {}
        for p in item_ids:
            W_dict[f'X_Item_{p}'] = (data['X_item_id'] == p).astype(int)
            W_dict[f'Y_Item_{p}'] = (data['Y_item_id'] == p).astype(int)
        for attr in attr_codes_tuples:
            attr_str = str(attr)
            W_dict[f'X_Attr_{attr_str}'] = (data['X_attr_ij'] == attr_str).astype(int)
            W_dict[f'Y_Attr_{attr_str}'] = (data['Y_attr_kl'] == attr_str).astype(int)
        W = pd.DataFrame(W_dict, index=self.exog_final.index)

        self.y_data = self.dependent.values
        self.W_data = W.values
        print(f"W_data shape: {self.W_data.shape}")

    # =========================================================================
    # Utility Calculation Functions
    # =========================================================================
    
    def _calculate_utility_z(self, W_data, packed_params):
        """
        Calculate the latent utility variable z for all observations.
        
        z = (1 + lambda) * V(X) - V(Y) + eta
        
        Where V(.) depends on robust_model:
        - Model 1: V = exp(alpha + beta)
        - Model 2: V = exp(alpha) + exp(beta)
        - Model 3: V = exp(alpha + beta) + gamma * (exp(alpha) + exp(beta))
        """
        theta = np.array(packed_params)
        lambda_val = theta[0]
        eta = theta[1]
        
        # Unpack alpha parameters (alpha_1 = 0 as reference)
        alphas = np.zeros(self.num_items)
        alphas[1:] = theta[2:2 + self.num_items - 1]
        
        # Unpack beta parameters (beta_11 = 0 as reference)
        betas = np.zeros(self.num_attr_combos)
        if self.robust_model == 3:
            betas[1:] = theta[2 + self.num_items - 1:-1]
            gamma = theta[-1]
        else:
            betas[1:] = theta[2 + self.num_items - 1:]
        
        # Column indices for W_data
        # X_Item: 0, 2, 4, ... (even indices up to 2*num_items)
        # Y_Item: 1, 3, 5, ... (odd indices up to 2*num_items)
        # X_Attr: 2*num_items, 2*num_items+2, ...
        # Y_Attr: 2*num_items+1, 2*num_items+3, ...
        
        X_item_cols = range(0, self.num_items * 2 - 1, 2)
        Y_item_cols = range(1, self.num_items * 2, 2)
        X_attr_cols = range(2 * self.num_items, 2 * self.num_items + self.num_attr_combos * 2 - 1, 2)
        Y_attr_cols = range(2 * self.num_items + 1, 2 * self.num_items + self.num_attr_combos * 2, 2)
        
        if self.robust_model == 1:
            # V = exp(alpha + beta)
            u_X = W_data[:, X_item_cols] @ alphas + W_data[:, X_attr_cols] @ betas
            u_Y = W_data[:, Y_item_cols] @ alphas + W_data[:, Y_attr_cols] @ betas
            z = (1 + lambda_val) * np.exp(u_X) - np.exp(u_Y) + eta
            
        elif self.robust_model == 2:
            # V = exp(alpha) + exp(beta)
            u_X = np.exp(W_data[:, X_item_cols] @ alphas) + np.exp(W_data[:, X_attr_cols] @ betas)
            u_Y = np.exp(W_data[:, Y_item_cols] @ alphas) + np.exp(W_data[:, Y_attr_cols] @ betas)
            z = (1 + lambda_val) * u_X - u_Y + eta
            
        elif self.robust_model == 3:
            # V = exp(alpha + beta) + gamma * (exp(alpha) + exp(beta))
            u_X_0 = np.exp(W_data[:, X_item_cols] @ alphas + W_data[:, X_attr_cols] @ betas)
            u_Y_0 = np.exp(W_data[:, Y_item_cols] @ alphas + W_data[:, Y_attr_cols] @ betas)
            u_X_1 = np.exp(W_data[:, X_item_cols] @ alphas) + np.exp(W_data[:, X_attr_cols] @ betas)
            u_Y_1 = np.exp(W_data[:, Y_item_cols] @ alphas) + np.exp(W_data[:, Y_attr_cols] @ betas)
            u_X = u_X_0 + gamma * u_X_1
            u_Y = u_Y_0 + gamma * u_Y_1
            z = (1 + lambda_val) * u_X - u_Y + eta
        
        return z
    
    def _calculate_y_hat(self, z):
        """Convert latent z to predicted probability."""
        if self.T > 0:
            return 1 / (1 + np.exp(-z / self.T))
        return (z > 0) * 1.0

    # =========================================================================
    # Model Functions for curve_fit
    # =========================================================================
    
    def model_function_nonlinear_curvefit(self, W_data, *packed_params):
        """Prediction function for NLS estimation using curve_fit."""
        z = self._calculate_utility_z(W_data, packed_params)
        return self._calculate_y_hat(z)
    
    def initial_fit(self, W_data, *packed_params):
        """
        Initial fit function without lambda and eta.
        Used for getting initial parameter estimates.
        """
        theta = np.array(packed_params)
        alphas = np.zeros(self.num_items)
        alphas[1:] = theta[:self.num_items - 1]
        betas = np.zeros(self.num_attr_combos)
        
        if self.robust_model == 3:
            betas[1:] = theta[self.num_items - 1:-1]
        else:
            betas[1:] = theta[self.num_items - 1:]
        
        X_item_cols = range(0, self.num_items * 2 - 1, 2)
        Y_item_cols = range(1, self.num_items * 2, 2)
        X_attr_cols = range(2 * self.num_items, 2 * self.num_items + self.num_attr_combos * 2 - 1, 2)
        Y_attr_cols = range(2 * self.num_items + 1, 2 * self.num_items + self.num_attr_combos * 2, 2)
        
        u_X = W_data[:, X_item_cols] @ alphas + W_data[:, X_attr_cols] @ betas
        u_Y = W_data[:, Y_item_cols] @ alphas + W_data[:, Y_attr_cols] @ betas
        z = np.exp(u_X) - np.exp(u_Y)
        
        return self._calculate_y_hat(z)
    
    def first_two_fit(self, W_data, *packed_params):
        """Fit function for lambda and eta only, with other params fixed."""
        theta = np.array(packed_params)
        lambda_val = theta[0]
        eta = theta[1]
        
        alphas = np.zeros(self.num_items)
        alphas[1:] = self.initial_other_params[:self.num_items - 1]
        betas = np.zeros(self.num_attr_combos)
        
        if self.robust_model == 3:
            betas[1:] = self.initial_other_params[self.num_items - 1:-1]
        else:
            betas[1:] = self.initial_other_params[self.num_items - 1:]
        
        X_item_cols = range(0, self.num_items * 2 - 1, 2)
        Y_item_cols = range(1, self.num_items * 2, 2)
        X_attr_cols = range(2 * self.num_items, 2 * self.num_items + self.num_attr_combos * 2 - 1, 2)
        Y_attr_cols = range(2 * self.num_items + 1, 2 * self.num_items + self.num_attr_combos * 2, 2)
        
        u_X = W_data[:, X_item_cols] @ alphas + W_data[:, X_attr_cols] @ betas
        u_Y = W_data[:, Y_item_cols] @ alphas + W_data[:, Y_attr_cols] @ betas
        z = (1 + lambda_val) * np.exp(u_X) - np.exp(u_Y) + eta
        
        return self._calculate_y_hat(z)

    # =========================================================================
    # Callback Functions for Iteration Tracking
    # =========================================================================
    
    def iteration_callback_ux(self, current_params):
        """Callback for tracking initial fit iterations."""
        current_iter = len(self.iteration_history) + 1
        y_hat = self.initial_fit(self.W_data, *current_params)
        residuals = self.y_data - y_hat
        rss = np.sum(residuals ** 2)
        
        self.iteration_history.append({
            'Iteration': current_iter,
            'RSS': rss,
            'beta_1,2': current_params[-8],
            'beta_1,3': current_params[-7],
            'beta_2,1': current_params[-6],
            'beta_2,2': current_params[-5],
            'beta_2,3': current_params[-4],
            'beta_3,1': current_params[-3],
            'beta_3,2': current_params[-2],
            'beta_3,3': current_params[-1],
            'params': current_params.copy()
        })
    
    def iteration_callback_with_plot(self, current_params):
        """Callback for tracking NLS iterations with optional plotting."""
        current_iter = len(self.iteration_history) + 1
        y_hat = self.model_function_nonlinear_curvefit(self.W_data, *current_params)
        residuals = self.y_data - y_hat
        rss = np.sum(residuals ** 2)
        
        if current_iter > 1:
            prev_params = self.iteration_history[-1]['params']
            param_step_norm = np.linalg.norm(current_params - prev_params)
        else:
            param_step_norm = np.nan
        
        self.iteration_history.append({
            'Iteration': current_iter,
            'RSS': rss,
            'lambda': current_params[0],
            'eta': current_params[1],
            'gamma': current_params[-1] if self.robust_model == 3 else np.nan,
            'Param_Step_Norm': param_step_norm,
            'params': current_params.copy()
        })
        
        print(f"Iteration {current_iter}: RSS={rss:.6f}, Lambda={current_params[0]:.4f}, "
              f"Eta={current_params[1]:.4f}, Step={param_step_norm:.6e}")
        
        # Save plots at key iterations
        if current_iter <= 3 or current_iter % 5 == 0:
            plot_dir = self._get_output_path('Training_Iteration_Plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            plt.figure(figsize=(6, 6))
            plt.scatter(y_hat, self.y_data, alpha=0.1, s=5, 
                       label=f'Data Points (N={len(self.y_data)})')
            plt.plot([0, 1], [0, 1], 'r--', label='y = x (Perfect Fit)')
            plt.title(f'Iteration {current_iter}: Predicted vs True Y (RSS={rss:.2f})')
            plt.xlabel('Predicted Y')
            plt.ylabel('True Y')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            
            plot_path = os.path.join(plot_dir, f'iter_{current_iter:03d}_yhat_vs_y.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"   Figure saved: {plot_path}")

    # =========================================================================
    # OLS and NLS Estimation
    # =========================================================================
    
    def OLS(self, data_clean, dependent, exog_final):
        """Run Pooled OLS estimation with clustered standard errors."""
        data_clean = data_clean.copy()
        data_clean['cluster_id'] = data_clean.index.get_level_values('pair_id')
        clusters = data_clean['cluster_id']
        
        mod = PooledOLS(dependent, exog_final, check_rank=False)
        ols_res = mod.fit(cov_type='clustered', clusters=clusters, debiased=True)
        return ols_res
    
    def NLS(self, fit_function, callback_function, initial_params):
        """
        Run Nonlinear Least Squares estimation using scipy curve_fit.
        
        Note: curve_fit does not support callbacks, so callback_function is ignored.
        """
        if callback_function is not None:
            lower = [-1, -200.0] + [-np.inf] * (len(initial_params) - 2)
            upper = [200.0, 200.0] + [np.inf] * (len(initial_params) - 2)
        else:
            lower = [-np.inf] * len(initial_params)
            upper = [np.inf] * len(initial_params)
        
        popt, pcov = curve_fit(
            f=fit_function,
            xdata=self.W_data,
            ydata=self.y_data,
            p0=initial_params,
            bounds=(lower, upper),
            method='trf'
        )
        return popt, pcov

    # =========================================================================
    # Parameter Initialization
    # =========================================================================
    
    def initialize_parameters(self):
        """Initialize parameters based on the specified starting method."""
        
        if self.robust_model == 3:
            self.num_total_params = self.num_items + 10  # +1 for gamma
            ols_res = self.OLS(self.data_clean, self.dependent, self.exog_final)
            initial = ols_res.params.tolist()
            self.initial_gamma = 0.0
            initial_other_params = np.array(initial[1:])  # (n-1) alphas + 8 betas
            self.initial_params = np.array([0.0, 0.0, *initial_other_params, self.initial_gamma])
        else:
            self.num_total_params = self.num_items + 9  # lambda + eta + (n-1) alphas + 8 betas
            
            if self.starting == 'ols':
                ols_res = self.OLS(self.data_clean, self.dependent, self.exog_final)
                initial = ols_res.params.tolist()
                initial_other_params = np.array(initial[1:])
                self.initial_params = np.array([0.0, 0.0, *initial_other_params])
                
            elif self.starting == 'zero':
                initial_other_params = np.zeros(self.num_total_params - 2)
                self.initial_params = np.array([0.0, 0.0, *initial_other_params])
                
            elif self.starting == 'prior':
                # Try to load from previous estimation results
                for model_type in ['A', 'B', 'C']:
                    try:
                        baseline_csv = self._get_model_filename(model_type).replace(
                            self._get_output_dir(), f'baseline/{self.Model_name}/Model_{self.robust_model}'
                        )
                        baseline_coeffs = pd.read_csv(baseline_csv)['Estimate'].tolist()
                        self.initial_params = np.array(baseline_coeffs)
                        break
                    except:
                        continue
                        
            elif self.starting == 'nls':
                # Two-stage NLS initialization
                self.iteration_history = []
                initial_params = np.zeros(self.num_total_params - 2)
                popt, _ = self.NLS(self.initial_fit, self.iteration_callback_ux, initial_params)
                
                self.initial_other_params = popt
                initial_params = np.array([0.0, 0.0])
                self.popt, self.pcov = self.NLS(self.first_two_fit, None, initial_params)
                self.initial_params = np.array([*self.popt, *self.initial_other_params])
                print(f"   Initial parameters: {self.initial_params}")

    # =========================================================================
    # Model A: NLS Estimation
    # =========================================================================
    
    def ModelANLS(self):
        """
        Run Model A: Nonlinear Least Squares estimation.
        
        Estimates all parameters jointly using curve_fit.
        """
        if not self.is_two:
            self.iteration_history = []
            self.popt, self.pcov = self.NLS(
                self.model_function_nonlinear_curvefit,
                self.iteration_callback_with_plot,
                self.initial_params
            )
        else:
            self.initial_other_params = self.initial_params[2:]
            initial_params = np.array([0.0, 0.0])
            self.popt, self.pcov = self.NLS(self.first_two_fit, None, initial_params)
            self.popt = np.array([*self.popt, *self.initial_other_params])
        
        print("\n--- curve_fit ---")
        
        # Calculate standard errors from covariance matrix
        standard_errors = np.sqrt(np.diag(self.pcov))
        
        # Build results DataFrame
        param_names = self._get_param_names()
        self.results_df = pd.DataFrame({
            'Parameter': pd.Series(param_names),
            'Estimate': pd.Series(self.popt),
            'Std. Err.': pd.Series(standard_errors),
            'Variance': pd.Series(np.diag(self.pcov))
        }).fillna(0)
        
        # Calculate latent variable statistics
        self._calculate_z_statistics()
        
        # Save results
        output_path = self._get_output_path(f'{self.Model_name}_NLS_estimation_T1(Model A).csv')
        self.results_df.to_csv(output_path, index=False, float_format='%.8f')
        print(f"\n✅ Results saved to: {output_path}")
        
        # Print key parameter variances
        print("\nKey parameter variances:")
        print(f"lambda variance: {self.results_df.iloc[0]['Variance']:.6e}")
        print(f"eta (const) variance: {self.results_df.iloc[1]['Variance']:.6e}")
        print(f"Covariance matrix shape: {self.pcov.shape}")
        
        # Save iteration history
        self._save_iteration_history('Training_Iteration_Plots_Model_A', 'NLS_Convergence_History.csv')
    
    def _calculate_z_statistics(self):
        """Calculate latent variable z statistics for model diagnostics."""
        z_hat = self._calculate_utility_z(self.W_data, self.popt)
        
        print("\n--- Latent Variable (z_hat) Statistics ---")
        print(f"z_hat mean: {np.mean(z_hat):.4f}")
        print(f"z_hat std: {np.std(z_hat):.4f}")
        
        try:
            kde = gaussian_kde(z_hat)
            self.f_u_0_hat = kde(0)[0]
            self.variance_z_hat = np.var(z_hat, ddof=0)
            
            print(f"✅ Var(U): {self.variance_z_hat:.6f}")
            print(f"✅ f_U(0): {self.f_u_0_hat:.6f}")
        except Exception as e:
            print(f"❌ Kernel density estimation failed: {e}")
    
    def _save_iteration_history(self, subdir, filename):
        """Save iteration history to CSV file."""
        try:
            history_df = pd.DataFrame(self.iteration_history)
            if 'params' in history_df.columns:
                history_df = history_df.drop(columns=['params'])
            
            history_dir = self._get_output_path(subdir)
            os.makedirs(history_dir, exist_ok=True)
            history_path = os.path.join(history_dir, filename)
            history_df.to_csv(history_path, index=False, float_format='%.8f')
            print(f"\n✅ Iteration history saved to: {history_path}")
        except:
            pass

    # =========================================================================
    # Model B: SMS with Bootstrap
    # =========================================================================
    
    def ModelBSMSBootstrap(self, B_runs=500):
        """
        Run Model B: Smoothed Maximum Score with Bootstrap standard errors.
        
        Requires Model A to be run first for initial parameters.
        """
        self.ModelANLS()
        
        def K_function(t):
            return logistic.cdf(t)
        
        def sms_objective(packed_params, y_data, W_data, h_N):
            z = self._calculate_utility_z(W_data, packed_params)
            K_z_h = K_function(z / h_N)
            return np.sum((y_data - K_z_h) ** 2)
        
        def run_sms_estimation(y_data_run, W_data_run, h_N_value, initial_params):
            lower_bounds = [-1.0] + [-np.inf] * (len(initial_params) - 2) + [0.0]
            upper_bounds = [np.inf] * len(initial_params)
            bounds = list(zip(lower_bounds, upper_bounds))
            
            initial_loss = sms_objective(initial_params, y_data_run, W_data_run, h_N_value)
            print(f"Initial Loss L(theta_0): {initial_loss:.8f}")
            
            try:
                result = minimize(
                    fun=sms_objective,
                    x0=initial_params,
                    args=(y_data_run, W_data_run, h_N_value),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'disp': True, 'maxiter': 1000, 'ftol': 1e-6, 'maxfun': 50000}
                )
                if result.success:
                    print(f"Final loss L(theta_hat): {result.fun:.8f}")
                    print(f"Iterations: {result.nit}")
                    return result.x
                else:
                    print(f"Minimize failed: {result.message}")
                    return None
            except Exception:
                return None
        
        def bootstrap_sms(y_data, W_data, h_N_value, initial_params, B=200, n_jobs=-1):
            N_obs = len(y_data)
            base_indices = np.arange(N_obs // 2)
            
            indices_list = []
            for _ in range(B):
                waiting_list = np.random.choice(base_indices, N_obs // 2, replace=True)
                indices_list.append(np.concatenate((waiting_list * 2, waiting_list * 2 + 1)))
            
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(run_sms_estimation)(
                    y_data[indices], W_data[indices], h_N_value, initial_params
                )
                for indices in indices_list
            )
            
            valid_results = np.array([r for r in results_list if r is not None])
            if valid_results.size == 0:
                return None, None
            
            variance_matrix = np.cov(valid_results, rowvar=False)
            standard_errors = np.sqrt(np.diag(variance_matrix))
            return variance_matrix, standard_errors
        
        # Main execution
        self.SMS_initial_params = self.popt
        N_obs = len(self.y_data)
        h_N_value = np.sqrt(self.variance_z_hat) * (N_obs) ** (-1/5)
        
        print(f"Bandwidth constant (std of z_hat): {np.sqrt(self.variance_z_hat):.4f}")
        print(f"h_N: {h_N_value:.4f}")
        
        # Step 1: Main SMS estimation
        print("--- Step 1: Main SMS Estimation ---")
        sms_main_result = run_sms_estimation(
            self.y_data, self.W_data, h_N_value, self.SMS_initial_params
        )
        
        if sms_main_result is None:
            print("Main SMS estimation failed, cannot continue Bootstrap.")
            return
        
        estimated_params = sms_main_result
        
        # Step 2: Bootstrap for standard errors
        print(f"\n--- Step 2: {B_runs} Bootstrap replications for std err ---")
        variance_matrix_sms, standard_errors_sms = bootstrap_sms(
            self.y_data, self.W_data, h_N_value, estimated_params, B=B_runs, n_jobs=-1
        )
        
        # Step 3: Save results
        if standard_errors_sms is not None:
            results_df = pd.DataFrame({
                'Parameter': self._get_param_names(),
                'Estimate': estimated_params,
                'Variance (Bootstrap)': np.diag(variance_matrix_sms),
                'Std. Err. (Bootstrap)': standard_errors_sms
            })
            
            print("\n--- Final SMS Estimation (Bootstrap std err) ---")
            print(results_df.head(5))
            
            output_path = self._get_output_path(f'{self.Model_name}_sms_results_bootstrap(Model B).csv')
            results_df.to_csv(output_path, index=False, float_format='%.8f')
            print(f"\n✅ File saved at: {output_path}")
        else:
            print("\nBootstrap estimation failed to converge.")

    # =========================================================================
    # Model C: Maximum Likelihood Estimation
    # =========================================================================
    
    def ModelCMLE(self):
        """
        Run Model C: Maximum Likelihood Estimation (Logit).

        When starting='zero', starts from all zeros (no Model A dependency).
        Otherwise, runs Model A first for initial parameters.
        """
        self.iteration_history = []

        if self.starting == 'zero':
            # Start from all zeros -- no Model A dependency
            if self.robust_model == 3:
                self.num_total_params = self.num_items + 10
                self.initial_params = np.zeros(self.num_total_params)
            else:
                self.num_total_params = self.num_items + 9
                self.initial_params = np.zeros(self.num_total_params)
        else:
            self.ModelANLS()
            initial = self.popt

            if self.robust_model == 3:
                self.num_total_params = self.num_items + 10
                self.initial_gamma = 0.0
                initial_other_params = np.array(initial[2:-1])
                self.initial_params = np.array([0.0, 0.0, *initial_other_params, self.initial_gamma])
            else:
                self.num_total_params = self.num_items + 9
                initial_other_params = np.array(initial[2:])
                self.initial_params = np.array([-0.3, 2.0, *initial_other_params])
        
        def negative_log_likelihood(packed_params, y_data, W_data):
            z = self._calculate_utility_z(W_data, packed_params)
            P_y1 = logistic.cdf(z / self.T)
            P_y1_clipped = np.clip(P_y1, 1e-15, 1 - 1e-15)
            log_likelihood = np.sum(
                y_data * np.log(P_y1_clipped) + (1 - y_data) * np.log(1 - P_y1_clipped)
            )
            return -log_likelihood
        
        def iteration_callback_C(current_params):
            current_nll = negative_log_likelihood(current_params, self.y_data, self.W_data)
            iteration = len(self.iteration_history)
            
            self.iteration_history.append({
                'iteration': iteration,
                'nll_value': current_nll,
                'lambda': current_params[0],
                'eta': current_params[1],
            })
            
            if iteration % 2 == 0:
                print(f"Callback Iteration {iteration}: NLL = {current_nll:.8f} | "
                      f"Lambda = {current_params[0]:.4f} | Eta = {current_params[1]:.4f}")
        
        lower_bounds = [-1.0] + [-np.inf] * (len(self.initial_params) - 2) + [0.0]
        upper_bounds = [np.inf] * len(self.initial_params)
        bounds = list(zip(lower_bounds, upper_bounds))
        
        print("--- Logit MLE Estimation ---")
        
        mle_result = minimize(
            fun=negative_log_likelihood,
            x0=self.initial_params,
            args=(self.y_data, self.W_data),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'maxiter': 5000, 'maxfun': 1000000},
            callback=iteration_callback_C
        )
        
        if mle_result.success:
            estimated_params = mle_result.x
            print("Computing numerical Hessian matrix...")
            
            hessian = approx_hess(
                estimated_params, negative_log_likelihood,
                args=(self.y_data, self.W_data)
            )
            
            try:
                covariance_matrix = np.linalg.inv(hessian)
                standard_errors = np.sqrt(np.diag(covariance_matrix))
                print("\n✅ MLE results computed successfully")
                
                results_df = pd.DataFrame({
                    'Parameter': self._get_param_names(),
                    'Estimate': estimated_params,
                    'Variance (MLE)': np.diag(covariance_matrix),
                    'Std. Err. (MLE)': standard_errors
                })
                
                print("\n--- Logit MLE Estimation (std err) ---")
                print(results_df[['Parameter', 'Estimate', 'Std. Err. (MLE)']].head(5))
                
                output_path = self._get_output_path(
                    f'{self.Model_name}_Logit_MLE_Results_Final(Model C).csv'
                )
                results_df.to_csv(output_path, index=False, float_format='%.8f')
                print(f"\n✅ File saved at: {output_path}")
                
                # Generate covariance heatmap
                self._plot_covariance_heatmap(covariance_matrix)
                
                # Run Wald test
                self._run_wald_test(estimated_params, covariance_matrix)
                
            except np.linalg.LinAlgError:
                print("❌ Hessian matrix is singular, cannot invert.")
        else:
            print(f"❌ Logit MLE estimation failed: {mle_result.message}")
    
    def _plot_covariance_heatmap(self, covariance_matrix):
        """Generate and save covariance matrix heatmap."""
        print("\nGenerating covariance matrix heatmap...")
        
        if self.robust_model == 3:
            selected_indices = [0, 1] + list(range(self.num_items + 1, self.num_items + 10))
            selected_names = ['lambda', 'eta', 'beta_1,2', 'beta_1,3', 'beta_2,1',
                            'beta_2,2', 'beta_2,3', 'beta_3,1', 'beta_3,2', 'beta_3,3', 'gamma']
        else:
            selected_indices = [0, 1] + list(range(self.num_items + 1, self.num_items + 9))
            selected_names = ['lambda', 'eta', 'beta_1,2', 'beta_1,3', 'beta_2,1',
                            'beta_2,2', 'beta_2,3', 'beta_3,1', 'beta_3,2', 'beta_3,3']
        
        sub_cov = covariance_matrix[np.ix_(selected_indices, selected_indices)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(sub_cov, annot=True, fmt=".4f", cmap='viridis',
                    xticklabels=selected_names, yticklabels=selected_names,
                    square=True, cbar_kws={"label": "Covariance Value"})
        plt.title(f'Covariance Matrix Heatmap - Model {self.robust_model}\n'
                  '(Diagonal = Variances)', fontsize=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        heatmap_path = self._get_output_path(f'Covariance_Heatmap_Model_{self.robust_model}.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"✅ Heatmap saved to: {heatmap_path}")
        
        print("\nDiagonal elements (Variances):")
        for name, var in zip(selected_names, np.diag(sub_cov)):
            print(f"{name:10s}: {var:.8f}")
    
    def _run_wald_test(self, estimated_params, covariance_matrix):
        """Run Wald joint test for H0: lambda = 0, eta = 0."""
        print("\nRunning Wald joint test (H0: lambda = 0, eta = 0)...")
        
        theta_sub = estimated_params[0:2]
        cov_sub = covariance_matrix[0:2, 0:2]
        
        try:
            inv_cov_sub = np.linalg.inv(cov_sub)
            wald_stat = theta_sub.T @ inv_cov_sub @ theta_sub
            df_test = 2
            p_value = 1 - chi2.cdf(wald_stat, df_test)
            
            print("-" * 50)
            print(" Wald Test Result (Joint Hypothesis):")
            print(f"  - Wald Statistic: {wald_stat:.4f}")
            print(f"  - Degrees of Freedom: {df_test}")
            print(f"  - P-value: {p_value:.4e}")
            
            if p_value < 0.01:
                conclusion = "At 1% significance, strongly reject H0 (lambda=0, eta=0)"
            elif p_value < 0.05:
                conclusion = "At 5% significance, reject H0"
            else:
                conclusion = "Cannot reject H0"
            print(f"  - Conclusion: {conclusion}")
            print("-" * 50)
            
            # Save test results
            test_path = self._get_output_path(f'Wald_Test_Model_{self.robust_model}.txt')
            with open(test_path, 'w') as f:
                f.write(f"Wald Joint Test for lambda=0 and eta=0\n")
                f.write(f"Wald Statistic: {wald_stat}\n")
                f.write(f"P-value: {p_value}\n")
                f.write(f"Result: {'Reject' if p_value < 0.05 else 'Fail to Reject'}\n")
                
        except np.linalg.LinAlgError:
            print("❌ Error: Covariance matrix is singular, cannot perform Wald test.")

    # =========================================================================
    # Results Presentation
    # =========================================================================
    
    def Present(self, which_model):
        """Present estimation results for the specified model."""
        output_path = self._get_model_filename(which_model)
        
        try:
            results_df = pd.read_csv(output_path, encoding='utf-8')
        except FileNotFoundError:
            print("Error: File not found. Please check the path and filename.")
            return
        except UnicodeDecodeError:
            print("Error: Encoding error. Try using 'gbk' encoding.")
            return
        
        print(f"\n--- Final Results for Model {which_model} ---")
        print(results_df.iloc[:2])
        
        if self.robust_model == 3:
            print(results_df.iloc[-1:])
            print(results_df.iloc[-9:-1])
        else:
            print(results_df.iloc[-8:])
        
        # Display beta matrix
        self._display_beta_matrix(results_df)
        
        # Plot alpha estimates
        self._plot_alpha_estimates(results_df, which_model)
    
    def _display_beta_matrix(self, results_df):
        """Display beta parameters as a 3x3 matrix."""
        if self.robust_model == 3:
            data_beta = [
                [0.0, results_df.iloc[-9]["Estimate"], results_df.iloc[-8]["Estimate"]],
                [results_df.iloc[-7]["Estimate"], results_df.iloc[-6]["Estimate"], results_df.iloc[-5]["Estimate"]],
                [results_df.iloc[-4]["Estimate"], results_df.iloc[-3]["Estimate"], results_df.iloc[-2]["Estimate"]]
            ]
        else:
            data_beta = [
                [0.0, results_df.iloc[-8]["Estimate"], results_df.iloc[-7]["Estimate"]],
                [results_df.iloc[-6]["Estimate"], results_df.iloc[-5]["Estimate"], results_df.iloc[-4]["Estimate"]],
                [results_df.iloc[-3]["Estimate"], results_df.iloc[-2]["Estimate"], results_df.iloc[-1]["Estimate"]]
            ]
        
        beta_df = pd.DataFrame(data_beta, index=[1, 2, 3], columns=[1, 2, 3])
        print("\nbeta:")
        print(beta_df.round(2))
    
    def _plot_alpha_estimates(self, results_df, which_model):
        """Plot alpha (item utility) estimates as a bar chart."""
        utility_item_1 = 0.0
        remaining_utilities = results_df.iloc[2:self.num_items + 1]["Estimate"].tolist()
        all_utilities = [utility_item_1] + remaining_utilities
        item_indices = [f"Item {i}" for i in range(1, self.num_items + 1)]
        
        df_utilities = pd.DataFrame({'Item': item_indices, 'Utility': all_utilities})
        df_sorted = df_utilities.sort_values(by='Utility', ascending=True).reset_index(drop=True)
        
        plt.figure(figsize=(20, 8))
        plt.bar(df_sorted['Item'], df_sorted['Utility'], color='skyblue')
        
        tick_positions = np.arange(0, len(df_sorted), 1)
        plt.xticks(tick_positions, df_sorted['Item'].iloc[tick_positions], rotation=90, fontsize=8)
        
        plt.title(f'{self.Model_name} Alpha Estimation Model {which_model}', fontsize=16)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Utility', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = self._get_output_path(f'{self.Model_name}-alpha-estimation-Model-{which_model}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"File saved at '{output_path}'")

    # =========================================================================
    # Utility Calculation and Delta
    # =========================================================================
    
    def calculate_utility_of_each_goods(self, which_model):
        """Calculate utility for each good (item x attribute combination)."""
        output_path = self._get_model_filename(which_model)
        
        try:
            self.results_df = pd.read_csv(output_path, encoding='utf-8')
        except FileNotFoundError:
            print("Error: File not found.")
            return
        
        estimates = self.results_df["Estimate"].values
        gamma = estimates[self.num_items + 9] if self.robust_model == 3 else None
        Utility = []
        for X in range(self.num_items):
            alpha = 0 if X == 0 else estimates[X + 1]
            exp_alpha = np.exp(alpha)
            for i in range(3):
                for j in range(3):
                    beta = 0 if (i == 0 and j == 0) else estimates[self.num_items + i * 3 + j]
                    exp_beta = np.exp(beta)
                    if self.robust_model == 1:
                        u = np.exp(alpha + beta)
                    elif self.robust_model == 2:
                        u = exp_alpha + exp_beta
                    elif self.robust_model == 3:
                        u = np.exp(alpha + beta) + gamma * (exp_alpha + exp_beta)
                    Utility.append(u)
        
        # Create DataFrame
        combinations = list(product(range(1, self.num_items + 1), range(1, 4), range(1, 4)))
        df = pd.DataFrame(combinations, columns=['index', 'attr_1', 'attr_2'])
        df['utility'] = Utility
        
        output_csv = self._get_output_path(f'{self.Model_name}_utility_of_each_goods_Model_{which_model}.csv')
        df.to_csv(output_csv, index=False, encoding='utf-8')
        self.U_goods = df
        
        # Plot utility distribution
        self._plot_utility_distribution(df, which_model)
    
    def _plot_utility_distribution(self, df, which_model):
        """Plot utility distribution as a bar chart."""
        item_indices = [f"Item {i}, Beta {j}, {k}" 
                       for i in range(1, self.num_items + 1) 
                       for j in range(1, 4) 
                       for k in range(1, 4)]
        
        df_utilities = pd.DataFrame({'Item': item_indices, 'Utility': df['utility'].tolist()})
        df_sorted = df_utilities.sort_values(by='Utility', ascending=True).reset_index(drop=True)
        
        plt.figure(figsize=(20, 8))
        plt.bar(df_sorted['Item'], df_sorted['Utility'], color='skyblue')
        
        tick_positions = np.arange(0, len(df_sorted), 5)
        plt.xticks(tick_positions, df_sorted['Item'].iloc[tick_positions], rotation=90, fontsize=8)
        
        plt.title(f'{self.Model_name} Utility of Goods in Model {which_model}', fontsize=16)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Utility', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = self._get_output_path(f'{self.Model_name}-utility-of-each-goods-Model-{which_model}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"File saved at '{output_path}'")
    
    def calculate_delta_and_delta_tilder(self, which_model):
        """
        Calculate delta (with bias) and delta_tilde (without bias) for each observation.
        
        delta = eta + (1 + lambda) * U(X) - U(Y)    [with bias]
        delta_tilde = U(X) - U(Y)                    [without bias / counterfactual]
        """
        lambda_val = self.results_df["Estimate"][0]
        eta = self.results_df["Estimate"][1]
        utility_lookup = self.U_goods['utility'].to_dict()

        for i in range(len(self.output_data_X)):
            current_X = self.output_data_X[i]
            current_Y = self.output_data_Y[i]

            id_X = current_X['X_num'] * 9 + current_X['attr'][0] * 3 + current_X['attr'][1]
            id_Y = current_Y['Y_num'] * 9 + current_Y['attr'][2] * 3 + current_Y['attr'][3]

            U_X = utility_lookup[id_X]
            U_Y = utility_lookup[id_Y]
            
            # Delta with bias
            current_X[f"Model_{which_model}_delta"] = round(eta + (1 + lambda_val) * U_X - U_Y, 4)
            current_Y[f"Model_{which_model}_delta"] = round(eta + (1 + lambda_val) * U_Y - U_X, 4)
            
            # Delta without bias (counterfactual)
            current_X[f"Model_{which_model}_delta_tilder"] = round(U_X - U_Y, 4)
            current_Y[f"Model_{which_model}_delta_tilder"] = round(U_Y - U_X, 4)
            
            self.output_data_X[i] = current_X
            self.output_data_Y[i] = current_Y
        
        # Save updated data
        with open(self._get_output_path(f'loss_aversion_X_for_Model_{which_model}.json'), 'w', encoding='utf-8') as f:
            json.dump(self.output_data_X, f, indent=4, ensure_ascii=False)
        with open(self._get_output_path(f'loss_aversion_Y_for_Model_{which_model}.json'), 'w', encoding='utf-8') as f:
            json.dump(self.output_data_Y, f, indent=4, ensure_ascii=False)

    # =========================================================================
    # Choice Probability Analysis
    # =========================================================================
    
    def _count_choices(self, x_values, y_values, comparison='discrete'):
        """
        Helper method to count choice combinations.
        
        Parameters:
        -----------
        x_values : list
            X perspective values (probabilities or delta values)
        y_values : list
            Y perspective values
        comparison : str
            'discrete' for yes/no classification based on prob > 0.5
            'delta' for delta > 0 classification
        """
        counts = {
            'yes_yes': 0, 'yes_tie': 0, 'yes_no': 0,
            'tie_yes': 0, 'tie_tie': 0, 'tie_no': 0,
            'no_yes': 0, 'no_tie': 0, 'no_no': 0
        }
        
        for x_val, y_val in zip(x_values, y_values):
            if comparison == 'discrete':
                # For probability tuples (yes_prob, no_prob)
                x_choice = 'yes' if x_val[0] > x_val[1] else ('tie' if x_val[0] == x_val[1] else 'no')
                y_choice = 'yes' if y_val[0] > y_val[1] else ('tie' if y_val[0] == y_val[1] else 'no')
            else:
                # For delta values
                x_choice = 'yes' if x_val < 0 else ('tie' if x_val == 0 else 'no')
                y_choice = 'yes' if y_val < 0 else ('tie' if y_val == 0 else 'no')
            
            counts[f'{x_choice}_{y_choice}'] += 1
        
        return counts
    
    def raw_choice_counts(self):
        """Calculate and display raw choice counts from LLM responses."""
        x_values = [item['Yes / No prob'] for item in self.output_data_X]
        y_values = [item['Yes / No prob'] for item in self.output_data_Y]
        
        counts = self._count_choices(x_values, y_values, comparison='discrete')
        
        # Calculate totals
        X_yes = counts['yes_yes'] + counts['yes_tie'] + counts['yes_no']
        X_tie = counts['tie_yes'] + counts['tie_tie'] + counts['tie_no']
        X_no = counts['no_yes'] + counts['no_tie'] + counts['no_no']
        Total = X_yes + X_tie + X_no
        
        print(f"\n{'='*60}")
        print(f"{self.feature} RAW CHOICE COUNTS for {self.Model_name}")
        print(f"{'='*60}")
        print(f"\nX responses: Yes={X_yes}, Tie={X_tie}, No={X_no}")
        
        Y_yes = counts['yes_yes'] + counts['tie_yes'] + counts['no_yes']
        Y_tie = counts['yes_tie'] + counts['tie_tie'] + counts['no_tie']
        Y_no = counts['yes_no'] + counts['tie_no'] + counts['no_no']
        print(f"Y responses: Yes={Y_yes}, Tie={Y_tie}, No={Y_no}")
        
        # Create DataFrames
        labels = ["Yes", "tie", "No"]
        number_data = [
            [counts['yes_yes'], counts['yes_tie'], counts['yes_no']],
            [counts['tie_yes'], counts['tie_tie'], counts['tie_no']],
            [counts['no_yes'], counts['no_tie'], counts['no_no']]
        ]
        fraction_data = (np.array(number_data) / Total * 100).tolist()
        
        number_df = pd.DataFrame(number_data, index=labels, columns=labels)
        fraction_df = pd.DataFrame(fraction_data, index=labels, columns=labels).round(2)
        
        print("\nNumbers (X \\ Y):")
        print(number_df)
        print("\nFraction (X \\ Y) %:")
        print(fraction_df)
        
        # Save to CSV
        save_dir = self._get_output_dir()
        number_df.to_csv(f'{save_dir}/{self.Model_name}_raw_choice_counts_number.csv')
        fraction_df.to_csv(f'{save_dir}/{self.Model_name}_raw_choice_counts_fraction.csv')
        print(f"\n✅ Saved to {save_dir}/")
        
        return number_df, fraction_df
    
    def choice_prob_from_model(self, which_model):
        """Calculate choice probabilities based on model-predicted delta values."""
        key_delta = f"Model_{which_model}_delta"
        key_tilde = f"Model_{which_model}_delta_tilder"
        x_delta, y_delta, x_delta_tilde, y_delta_tilde = [], [], [], []
        for x_item, y_item in zip(self.output_data_X, self.output_data_Y):
            x_delta.append(x_item[key_delta])
            y_delta.append(y_item[key_delta])
            x_delta_tilde.append(x_item[key_tilde])
            y_delta_tilde.append(y_item[key_tilde])

        # With bias
        counts = self._count_choices(x_delta, y_delta, comparison='delta')
        self._print_choice_matrix(counts, f"{self.feature} {self.Model_name} with bias")

        # Save with bias results
        self._save_choice_matrix(counts, which_model, with_bias=True)

        # Without bias (counterfactual)
        
        counts_nb = self._count_choices(x_delta_tilde, y_delta_tilde, comparison='delta')
        self._print_choice_matrix(counts_nb, f"{self.feature} {self.Model_name} (No Bias)")
    
    def _print_choice_matrix(self, counts, title):
        """Print choice count matrix."""
        Total = sum(counts.values())
        
        print(f"{self.feature} results of {title}:")
        
        labels = ["yes", "tie", "no"]
        number_data = [
            [counts['yes_yes'], counts['yes_tie'], counts['yes_no']],
            [counts['tie_yes'], counts['tie_tie'], counts['tie_no']],
            [counts['no_yes'], counts['no_tie'], counts['no_no']]
        ]
        fraction_data = (np.array(number_data) / Total * 100).tolist()
        
        number_df = pd.DataFrame(number_data, index=labels, columns=labels)
        fraction_df = pd.DataFrame(fraction_data, index=labels, columns=labels).round(2)
        
        print("\nNumbers of X/Y:")
        print(number_df)
        print("\nFraction of X/Y:")
        print(fraction_df)
        print()
    
    def _save_choice_matrix(self, counts, which_model, with_bias=True):
        """Save choice count matrix to CSV."""
        Total = sum(counts.values())
        labels = ["yes", "tie", "no"]
        
        number_data = [
            [counts['yes_yes'], counts['yes_tie'], counts['yes_no']],
            [counts['tie_yes'], counts['tie_tie'], counts['tie_no']],
            [counts['no_yes'], counts['no_tie'], counts['no_no']]
        ]
        fraction_data = (np.array(number_data) / Total * 100).tolist()
        
        number_df = pd.DataFrame(number_data, index=labels, columns=labels)
        fraction_df = pd.DataFrame(fraction_data, index=labels, columns=labels).round(2)
        
        suffix = '' if with_bias else '_no_bias'
        number_df.to_csv(self._get_output_path(
            f'{self.Model_name}_choice_prob_number_with_estimation_model_{which_model}{suffix}.csv'
        ))
        fraction_df.to_csv(self._get_output_path(
            f'{self.Model_name}_choice_prob_fraction_with_estimation_model_{which_model}{suffix}.csv'
        ))
    
    def ChoiceProb(self):
        """Legacy method for choice probability analysis with histogram generation."""
        x_values = [item['Yes / No prob'] for item in self.output_data_X]
        y_values = [item['Yes / No prob'] for item in self.output_data_Y]
        
        counts = self._count_choices(x_values, y_values, comparison='discrete')
        
        X_yes = counts['yes_yes'] + counts['yes_tie'] + counts['yes_no']
        X_tie = counts['tie_yes'] + counts['tie_tie'] + counts['tie_no']
        X_no = counts['no_yes'] + counts['no_tie'] + counts['no_no']
        Y_yes = counts['yes_yes'] + counts['tie_yes'] + counts['no_yes']
        Y_tie = counts['yes_tie'] + counts['tie_tie'] + counts['no_tie']
        Y_no = counts['yes_no'] + counts['tie_no'] + counts['no_no']
        
        print(f"{self.feature} results of {self.Model_name}:")
        print(f"Number of 'Yes' in X:{X_yes}; Number of 'Tie' in X:{X_tie}; Number of 'No' in X:{X_no}")
        Total = X_yes + X_tie + X_no
        print(f"Fraction of 'Yes' in X:{X_yes/Total*100:.2f}%; Fraction of 'Tie' in X:{X_tie/Total*100:.2f}%; Fraction of 'No' in X:{X_no/Total*100:.2f}%")
        print(f"Number of 'Yes' in Y:{Y_yes}; Number of 'Tie' in Y:{Y_tie}; Number of 'No' in Y:{Y_no}")
        print(f"Fraction of 'Yes' in Y:{Y_yes/Total*100:.2f}%; Fraction of 'Tie' in Y:{Y_tie/Total*100:.2f}%; Fraction of 'No' in Y:{Y_no/Total*100:.2f}%")
        
        self._print_choice_matrix(counts, self.Model_name)
        
        # Save results
        labels = ["yes", "tie", "no"]
        number_data = [
            [counts['yes_yes'], counts['yes_tie'], counts['yes_no']],
            [counts['tie_yes'], counts['tie_tie'], counts['tie_no']],
            [counts['no_yes'], counts['no_tie'], counts['no_no']]
        ]
        fraction_data = (np.array(number_data) / Total * 100).tolist()
        
        number_df = pd.DataFrame(number_data, index=labels, columns=labels)
        fraction_df = pd.DataFrame(fraction_data, index=labels, columns=labels).round(2)
        
        number_df.to_csv(f'{self.feature}/{self.Model_name}/{self.Model_name}_choice_prob_number.csv')
        fraction_df.to_csv(f'{self.feature}/{self.Model_name}/{self.Model_name}_choice_prob_fraction.csv')
        
        # Generate probability histograms
        self._generate_probability_histograms()
    
    def _generate_probability_histograms(self):
        """Generate probability distribution histograms."""
        no_prob_X = np.array([item['Yes / No prob'][1] for item in self.output_data_X])
        no_prob_Y = np.array([item['Yes / No prob'][1] for item in self.output_data_Y])
        
        dist_a = no_prob_X
        dist_b = no_prob_Y
        dist_c = 0.5 * no_prob_X + 0.5 * no_prob_Y
        
        histogram_dir = f'{self.feature}/{self.Model_name}/Histogram_Prob'
        os.makedirs(histogram_dir, exist_ok=True)
        
        # Stacked histogram
        TOTAL_SAMPLE_SIZE = len(dist_c)
        condition_red = (no_prob_X > 0.5) & (no_prob_Y > 0.5)
        condition_blue = (no_prob_X < 0.5) & (no_prob_Y < 0.5)
        condition_gray = np.isclose(no_prob_X, 0.5) | np.isclose(no_prob_Y, 0.5)
        condition_others = ~(condition_red | condition_blue | condition_gray)
        
        data_red = dist_c[condition_red]
        data_blue = dist_c[condition_blue]
        data_gray = dist_c[condition_gray]
        data_white = dist_c[condition_others]
        
        num_bins = 51
        bins = np.linspace(0, 1, num_bins + 1)
        bin_width = bins[1] - bins[0]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        freq_red, _ = np.histogram(data_red, bins=bins)
        freq_blue, _ = np.histogram(data_blue, bins=bins)
        freq_gray, _ = np.histogram(data_gray, bins=bins)
        freq_white, _ = np.histogram(data_white, bins=bins)
        
        freq_red = freq_red / TOTAL_SAMPLE_SIZE
        freq_blue = freq_blue / TOTAL_SAMPLE_SIZE
        freq_gray = freq_gray / TOTAL_SAMPLE_SIZE
        freq_white = freq_white / TOTAL_SAMPLE_SIZE
        
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'gray', 'white']
        labels = [
            r'$P(\text{no|X}) > 0.5 \text{ and } P(\text{no|Y}) > 0.5$',
            r'$P(\text{no|X}) < 0.5 \text{ and } P(\text{no|Y}) < 0.5$',
            r'$P(\text{no|X}) \text{ or } P(\text{no|Y}) = 0.5$',
            'Remaining / Other Cases'
        ]
        
        bottoms = np.zeros(num_bins)
        for freq, color, label in zip([freq_red, freq_blue, freq_gray, freq_white], colors, labels):
            plt.bar(bin_centers, freq, width=bin_width, color=color, 
                   edgecolor='black', label=label, bottom=bottoms)
            bottoms += freq
        
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Threshold 0.5')
        plt.title(r'Distribution of $\frac{1}{2} P(\text{no|X}) + \frac{1}{2} P(\text{no|Y})$')
        plt.xlabel('Probability')
        plt.ylabel('Fraction of Samples')
        plt.xlim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Category Conditions', loc='best')
        plt.savefig(os.path.join(histogram_dir, 'distribution_c_stacked.png'))
        plt.close()
        
        # Individual histograms
        for dist, name, color in [(dist_a, 'a', 'skyblue'), (dist_b, 'b', 'salmon'), (dist_c, 'c', 'lightgreen')]:
            plt.figure(figsize=(8, 5))
            plt.hist(dist, bins=100, color=color, edgecolor='black', density=False)
            plt.title(f'Distribution of P(no|{"X" if name == "a" else "Y" if name == "b" else "combined"})')
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(histogram_dir, f'distribution_{name}_prob_diff.png'))
            plt.close()
            print(f"Histogram for distribution {name}) saved")
    
    def Choice_prob_logits(self):
        """Calculate choice probabilities using logit aggregation."""
        X_yes, X_no, Y_yes, Y_no = 0, 0, 0, 0
        yes_yes_pair, yes_no_pair, no_yes_pair, no_no_pair = 0, 0, 0, 0
        
        for i in range(len(self.output_data_X)):
            X_item = self.output_data_X[i]["Yes / No prob"]
            Y_item = self.output_data_Y[i]["Yes / No prob"]
            
            X_yes_temp, X_no_temp = X_item[0], X_item[1]
            Y_yes_temp, Y_no_temp = Y_item[0], Y_item[1]
            
            X_yes += X_yes_temp
            Y_yes += Y_yes_temp
            X_no += X_no_temp
            Y_no += Y_no_temp
            
            yes_yes_pair += X_yes_temp * Y_yes_temp
            yes_no_pair += X_yes_temp * Y_no_temp
            no_yes_pair += X_no_temp * Y_yes_temp
            no_no_pair += X_no_temp * Y_no_temp
        
        print(f"{self.feature} results of {self.Model_name} at temperature 1:")
        print(f"Number of 'Yes' in X:{X_yes}; Number of 'No' in X:{X_no}")
        print(f"Fraction of 'Yes' in X:{X_yes/(X_yes+X_no)*100:.2f}%; Fraction of 'No' in X:{X_no/(X_yes+X_no)*100:.2f}%")
        print(f"Number of 'Yes' in Y:{Y_yes}; Number of 'No' in Y:{Y_no}")
        print(f"Fraction of 'Yes' in Y:{Y_yes/(Y_yes+Y_no)*100:.2f}%; Fraction of 'No' in Y:{Y_no/(Y_yes+Y_no)*100:.2f}%")
        print()
        print(f'Number of X yes and Y no: {yes_no_pair:.2f}')
        print(f'Number of X no and Y yes: {no_yes_pair:.2f}')
        print(f'Number of X no and Y no: {no_no_pair:.2f}')
        print(f'Number of X yes and Y yes: {yes_yes_pair:.2f}')
        print()
        Total = X_yes + X_no
        print(f'Fraction of X yes and Y no: {yes_no_pair/Total*100:.2f}%')
        print(f'Fraction of X no and Y yes: {no_yes_pair/Total*100:.2f}%')
        print(f'Fraction of X no and Y no: {no_no_pair/Total*100:.2f}%')
        print(f'Fraction of X yes and Y yes: {yes_yes_pair/Total*100:.2f}%')
