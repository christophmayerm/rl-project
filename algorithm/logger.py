from utils.tabular import *
from algorithm.model_chooser import *
from utils.tabular_operations import policy_equiv_check, model_equiv_check
from tqdm import tqdm
import numpy as np

class Logger(object):
    """
    Enhanced Logger with multiple progress bar options and metrics evaluation
    """

    def __init__(self, mdp, model_chooser, verbose=2, log_interval=100, 
                 use_wandb=False, wandb_config=None, progress_style='standard',
                 metrics_evaluator=None):
        """
        :param progress_style: Style of progress bar
            - 'standard': Default tqdm bar
            - 'rich': Rich progress bar (requires rich library)
            - 'dual': Dual bars for inner/outer loops (for SA-PMI)
            - 'minimal': Minimal percentage display
            - 'none': No progress bar
        :param metrics_evaluator: IterationMetricsEvaluator instance
        """
        self.mdp = mdp
        self.model_chooser = model_chooser
        self.verbose = verbose
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.progress_style = progress_style
        self.metrics_evaluator = metrics_evaluator

        # LOGGING ATTRIBUTES
        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.p_advantages = list()
        self.m_advantages = list()
        self.p_dist_sup = list()
        self.p_dist_mean = list()
        self.m_dist_sup = list()
        self.m_dist_mean = list()
        self.alfas = list()
        self.betas = list()
        self.w_target = list()
        self.w_current = list()
        self.p_change = list()
        self.m_change = list()
        self.bound = list()
        
        # Adversarial tracking
        self.adversarial_budgets = list()
        self.adversarial_disadvantages = list()
        self.adversarial_dist_sups = list()
        self.adversarial_dist_means = list()
        
        # Progress bar
        self.pbar = None
        self.max_iter = None
        
        # Wandb initialization
        self.wandb_run = None
        self.wandb = None
        if self.use_wandb:
            self._init_wandb(wandb_config)
    
    def _init_wandb(self, wandb_config):
        """Initialize Weights & Biases logging"""
        try:
            import wandb
            self.wandb = wandb
            
            if wandb_config is None:
                wandb_config = {}
            
            # Extract config
            project = wandb_config.get('project', 'spmi-experiments')
            name = wandb_config.get('name', None)
            config = wandb_config.get('config', {})
            tags = wandb_config.get('tags', [])
            group = wandb_config.get('group', None)
            job_type = wandb_config.get('job_type', None)
            
            # Initialize run
            self.wandb_run = self.wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                group=group,
                job_type=job_type,
                reinit=True
            )
            
            if self.verbose >= 1:
                print(f"✓ W&B initialized: {self.wandb_run.url}")
                
        except ImportError:
            print("⚠️  wandb not installed. Install with: pip install wandb")
            self.use_wandb = False
            self.wandb = None
        except Exception as e:
            print(f"⚠️  Failed to initialize wandb: {e}")
            self.use_wandb = False
            self.wandb = None
    
    def set_max_iter(self, max_iter):
        """Set maximum iterations and initialize progress bar"""
        self.max_iter = max_iter
        
        if self.progress_style == 'none':
            return
        
        if self.progress_style == 'standard':
            self._init_standard_progress()
        elif self.progress_style == 'rich':
            self._init_rich_progress()
        elif self.progress_style == 'dual':
            self._init_dual_progress()
        elif self.progress_style == 'minimal':
            self._init_minimal_progress()
    
    def _init_standard_progress(self):
        """Initialize standard tqdm progress bar"""
        self.pbar = tqdm(
            total=self.max_iter,
            desc="🔄 Training",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            ncols=120,
            leave=True,
            position=0
        )
    
    def _init_rich_progress(self):
        """Initialize rich progress bar (requires rich library)"""
        try:
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
            
            self.pbar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TextColumn("• J: {task.fields[performance]:.4f}"),
                TextColumn("• α: {task.fields[alpha]:.3f}"),
                TextColumn("• β: {task.fields[beta]:.3f}"),
            )
            self.pbar.start()
            self.pbar_task = self.pbar.add_task(
                "Training", 
                total=self.max_iter,
                performance=0.0,
                alpha=0.0,
                beta=0.0
            )
        except ImportError:
            print("⚠️  rich not installed. Falling back to standard progress bar.")
            print("   Install with: pip install rich")
            self._init_standard_progress()
    
    def _init_dual_progress(self):
        """Initialize dual progress bars for nested loops"""
        from tqdm import tqdm
        # Outer bar for overall progress
        self.pbar = tqdm(
            total=self.max_iter,
            desc="🎯 Overall",
            position=0,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
        )
        # Inner bar for detailed metrics
        self.pbar_inner = tqdm(
            total=100,
            desc="📊 Metrics",
            position=1,
            leave=False,
            bar_format='{desc}: J={postfix[0][J]:.4f} α={postfix[0][α]:.3f} β={postfix[0][β]:.3f}'
        )
    
    def _init_minimal_progress(self):
        """Initialize minimal progress (just percentage updates)"""
        self.pbar = None
        self.last_percent = -1

    def update(self, J_p_m, alfa_star, beta_star, p_er_adv, m_er_adv,
            p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
            target_policy, target_policy_old, target_model,
            target_model_old, convergence, bound,
            adversarial_budget=None,
            adversarial_disadvantage=None,
            adversarial_dist_sup=None,
            adversarial_dist_mean=None,
            # NEW: Pass precomputed distributions for metrics
            policy=None, model=None, reward=None,
            d_mu=None, delta_mu=None):
        """Update logger with current iteration metrics"""
        
        # Data collections
        self.iterations.append(self.iteration)
        self.evaluations.append(J_p_m)
        self.alfas.append(alfa_star)
        self.betas.append(beta_star)
        self.p_advantages.append(p_er_adv)
        self.m_advantages.append(m_er_adv)
        self.p_dist_sup.append(p_dist_sup)
        self.p_dist_mean.append(p_dist_mean)
        self.m_dist_sup.append(m_dist_sup)
        self.m_dist_mean.append(m_dist_mean)
        self.bound.append(bound)

        # Target policy change check
        if isinstance(target_policy, TabularPolicy):
            p_check_target = policy_equiv_check(target_policy, target_policy_old)
        else:
            p_check_target = np.nan
        self.p_change.append(p_check_target)
        
        # Target model change check
        if isinstance(target_model, TabularModel):
            m_check_target = model_equiv_check(target_model, target_model_old)
        else:
            m_check_target = np.nan
        self.m_change.append(m_check_target)

        # Collect adversarial data if provided
        is_adversarial = adversarial_budget is not None
        if is_adversarial:
            self.adversarial_budgets.append(adversarial_budget)
            self.adversarial_disadvantages.append(adversarial_disadvantage)
            self.adversarial_dist_sups.append(adversarial_dist_sup)
            self.adversarial_dist_means.append(adversarial_dist_mean)

        # NEW: Evaluate auxiliary metrics if evaluator is attached
        auxiliary_metrics = {}
        if self.metrics_evaluator is not None and policy is not None and model is not None:
            try:
                auxiliary_metrics = self.metrics_evaluator.evaluate(
                    policy=policy,
                    model=model,
                    reward=reward,
                    d_mu=d_mu,
                    delta_mu=delta_mu,
                    iteration=self.iteration
                )
            except Exception as e:
                if self.verbose >= 2:
                    print(f"⚠️  Metrics evaluation failed at iteration {self.iteration}: {e}")

        # Update progress bar based on style
        self._update_progress(J_p_m, alfa_star, beta_star, bound, 
                            adversarial_budget, is_adversarial)
        
        # Verbose output
        if self.verbose == 3:
            self._print_detailed(J_p_m, alfa_star, beta_star, p_er_adv, m_er_adv,
                                p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
                                convergence, bound, adversarial_budget, 
                                adversarial_disadvantage, adversarial_dist_sup,
                                adversarial_dist_mean, auxiliary_metrics)
        elif self.verbose == 2 and self.iteration % self.log_interval == 0:
            self._print_summary(J_p_m, alfa_star, beta_star, bound, 
                              adversarial_budget, is_adversarial, auxiliary_metrics)
        
        # Log to wandb
        if self.use_wandb and self.wandb is not None and self.wandb_run is not None:
            try:
                wandb_metrics = {
                    'performance': J_p_m,
                    'alpha': alfa_star,
                    'beta': beta_star,
                    'bound': bound,
                    'policy_er_advantage': p_er_adv if not np.isnan(p_er_adv) else 0,
                    'model_er_advantage': m_er_adv if not np.isnan(m_er_adv) else 0,
                    'policy_dist_sup': p_dist_sup if not np.isnan(p_dist_sup) else 0,
                    'policy_dist_mean': p_dist_mean if not np.isnan(p_dist_mean) else 0,
                    'model_dist_sup': m_dist_sup if not np.isnan(m_dist_sup) else 0,
                    'model_dist_mean': m_dist_mean if not np.isnan(m_dist_mean) else 0,
                }
                
                if is_adversarial:
                    wandb_metrics.update({
                        'adversarial_budget': adversarial_budget,
                        'adversarial_disadvantage': adversarial_disadvantage,
                        'adversarial_dist_sup': adversarial_dist_sup,
                        'adversarial_dist_mean': adversarial_dist_mean,
                    })
                
                # Add auxiliary metrics to wandb
                if auxiliary_metrics:
                    wandb_metrics.update({f'aux_{k}': v for k, v in auxiliary_metrics.items() 
                                        if k != 'iteration'})
                
                self.wandb.log(wandb_metrics, step=self.iteration)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"⚠️  W&B logging failed at iteration {self.iteration}: {e}")
                self.use_wandb = False

        # Model vector coefficients computation
        if isinstance(self.model_chooser, SetModelChooser):
            model_vector = self.mdp.model_vector
            model_set = self.model_chooser.model_set
            n_models = len(model_vector)
            if isinstance(target_model, TabularModel):
                for i in range(n_models):
                    if model_equiv_check(model_set[i], target_model):
                        target_index = i
                        break
                target_vector = np.zeros(n_models)
                target_vector[target_index] = 1
                new_model_vector = beta_star * target_vector + (1 - beta_star) * model_vector
                self.mdp.model_vector = new_model_vector

                if self.verbose == 3:
                    print('\ntarget_model: {0}'.format(target_vector))
                    print('current_model: {0}'.format(new_model_vector))
                    
                self.w_current.append(new_model_vector)
                self.w_target.append(target_vector)
            else:
                self.w_current.append(model_vector)
                target_vector = np.empty(n_models)
                target_vector[:] = np.nan
                self.w_target.append(target_vector)

        # Iteration update
        self.iteration = self.iteration + 1
    
    def _update_progress(self, J_p_m, alfa_star, beta_star, bound, 
                        adv_budget, is_adversarial):
        """Update progress bar based on style"""
        if self.progress_style == 'standard' and self.pbar is not None:
            postfix = {
                'J': f'{J_p_m:.4f}',
                'α': f'{alfa_star:.3f}',
                'β': f'{beta_star:.3f}',
                'bound': f'{bound:.3f}'
            }
            if is_adversarial:
                postfix['B'] = f'{adv_budget:.3f}'
            
            self.pbar.set_postfix(postfix)
            self.pbar.update(1)
            
        elif self.progress_style == 'rich' and self.pbar is not None:
            self.pbar.update(
                self.pbar_task,
                advance=1,
                performance=J_p_m,
                alpha=alfa_star,
                beta=beta_star
            )
            
        elif self.progress_style == 'dual' and self.pbar is not None:
            self.pbar.update(1)
            self.pbar_inner.set_postfix([{
                'J': J_p_m,
                'α': alfa_star,
                'β': beta_star
            }])
            
        elif self.progress_style == 'minimal':
            percent = int((self.iteration / self.max_iter) * 100)
            if percent != self.last_percent and percent % 10 == 0:
                print(f"Progress: {percent}% | J={J_p_m:.4f} | α={alfa_star:.3f} β={beta_star:.3f}")
                self.last_percent = percent
    
    def _print_detailed(self, J_p_m, alfa_star, beta_star, p_er_adv, m_er_adv,
                       p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
                       convergence, bound, adv_budget, adv_disadv, 
                       adv_dist_sup, adv_dist_mean, aux_metrics=None):
        """Print detailed metrics (verbose=3)"""
        if self.pbar is not None and self.progress_style == 'standard':
            self.pbar.clear()
        
        print('\n' + '='*60)
        print(f'Iteration: {self.iteration}')
        print('='*60)
        print('Performance: {0}'.format(J_p_m))
        print('Alpha/Beta: {0}/{1}'.format(alfa_star, beta_star))
        print('Bound: {0}'.format(bound))
        print('Convergence Condition: {0}\n'.format(convergence))

        print('Policy Advantage: {0}'.format(p_er_adv))
        print('Policy Dist Sup: {0}'.format(p_dist_sup))
        print('Policy Dist Mean: {0}'.format(p_dist_mean))

        print('\nModel Advantage: {0}'.format(m_er_adv))
        print('Model Dist Sup: {0}'.format(m_dist_sup))
        print('Model Dist Mean: {0}'.format(m_dist_mean))

        if adv_budget is not None:
            print('\n--- ADVERSARIAL INFO ---')
            print('Budget: {0}'.format(adv_budget))
            print('Disadvantage: {0}'.format(adv_disadv))
            print('Dist Sup: {0}'.format(adv_dist_sup))
            print('Dist Mean: {0}'.format(adv_dist_mean))
        
        if aux_metrics:
            print('\n--- AUXILIARY METRICS ---')
            for key, val in aux_metrics.items():
                if key != 'iteration':
                    print(f'{key}: {val:.4f}')
        
        if self.pbar is not None and self.progress_style == 'standard':
            self.pbar.refresh()
    
    def _print_summary(self, J_p_m, alfa_star, beta_star, bound, 
                      adv_budget, is_adversarial, aux_metrics=None):
        """Print summary (verbose=2)"""
        if self.pbar is not None and self.progress_style == 'standard':
            self.pbar.clear()
        
        summary = (f"[Iter {self.iteration:5d}] "
                  f"J={J_p_m:7.4f} | "
                  f"α={alfa_star:.3f} β={beta_star:.3f} | "
                  f"Bound={bound:7.3f}")
        
        if is_adversarial:
            summary += f" | B={adv_budget:.3f}"
        
        if aux_metrics:
            # Add coverage and entropy to summary
            if 'state_coverage' in aux_metrics:
                summary += f" | Cov={aux_metrics['state_coverage']:.3f}"
            if 'model_tv_mean' in aux_metrics:
                summary += f" | TV={aux_metrics['model_tv_mean']:.3f}"
        
        print(summary)
        
        if self.pbar is not None and self.progress_style == 'standard':
            self.pbar.refresh()
    
    def print_final_summary(self):
        """Print final summary at end of training"""
        if self.verbose >= 1:
            print("\n" + "="*70)
            print("TRAINING COMPLETED")
            print("="*70)
            print(f"Total Iterations:    {self.iteration}")
            print(f"Final Performance:   {self.evaluations[-1]:.6f}")
            print(f"Initial Performance: {self.evaluations[0]:.6f}")
            print(f"Improvement:         {self.evaluations[-1] - self.evaluations[0]:.6f}")
            
            if len(self.adversarial_budgets) > 0:
                print(f"\nFinal Adversarial Budget: {self.adversarial_budgets[-1]:.6f}")
            
            if self.metrics_evaluator and self.metrics_evaluator.history:
                final_metrics = self.metrics_evaluator.history[-1]
                print(f"\nFinal Auxiliary Metrics:")
                for key, val in final_metrics.items():
                    if key != 'iteration':
                        print(f"  {key}: {val:.4f}")
            
            print("="*70)

    def reset(self):
        """Reset all logging attributes"""
        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.p_advantages = list()
        self.m_advantages = list()
        self.p_dist_sup = list()
        self.p_dist_mean = list()
        self.m_dist_sup = list()
        self.m_dist_mean = list()
        self.alfas = list()
        self.betas = list()
        self.w_current = list()
        self.p_change = list()
        self.m_change = list()
        self.w_target = list()
        self.bound = list()
        
        # Reset adversarial tracking
        self.adversarial_budgets = list()
        self.adversarial_disadvantages = list()
        self.adversarial_dist_sups = list()
        self.adversarial_dist_means = list()
        
        # Reset metrics evaluator
        if self.metrics_evaluator is not None:
            self.metrics_evaluator.reset()
        
        # Close progress bar but NOT wandb
        if self.pbar is not None:
            if self.progress_style == 'rich':
                self.pbar.stop()
            elif self.progress_style == 'dual':
                self.pbar.close()
                if hasattr(self, 'pbar_inner'):
                    self.pbar_inner.close()
            else:
                self.pbar.close()
            self.pbar = None
    
    def close(self):
        """Close progress bar and wandb run"""
        if self.pbar is not None:
            if self.progress_style == 'rich':
                self.pbar.stop()
            elif self.progress_style == 'dual':
                self.pbar.close()
                if hasattr(self, 'pbar_inner'):
                    self.pbar_inner.close()
            else:
                self.pbar.close()
            self.pbar = None
        
        if self.use_wandb and self.wandb is not None and self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                if self.verbose >= 1:
                    print(f"✓ W&B run finished: {self.wandb_run.url}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"⚠️  Error finishing W&B run: {e}")
            finally:
                self.wandb_run = None

    def save(self, dir_path, file_name, entries=None):
        """Save execution data to CSV file"""
        header_string = 'iterations;evaluations;p_advantages;m_advantages;' \
                        'p_dist_sup;p_dist_mean;m_dist_sup;m_dist_mean;alfa;beta;p_change;m_change;bound'

        execution_data = [self.iterations, self.evaluations,
                          self.p_advantages, self.m_advantages,
                          self.p_dist_sup, self.p_dist_mean,
                          self.m_dist_sup, self.m_dist_mean,
                          self.alfas, self.betas, self.p_change,
                          self.m_change, self.bound]
        
        # Add adversarial data if available
        if len(self.adversarial_budgets) > 0:
            header_string += ';adv_budget;adv_disadvantage;adv_dist_sup;adv_dist_mean'
            execution_data.extend([
                self.adversarial_budgets,
                self.adversarial_disadvantages,
                self.adversarial_dist_sups,
                self.adversarial_dist_means
            ])
            
        if isinstance(self.model_chooser, SetModelChooser):
            if len(self.model_chooser.model_set) == 2:
                header_string = header_string + ';w_current[0];w_current[1];w_target[0];w_target[1]'

                current = np.array(self.w_current)
                target = np.array(self.w_target)

                execution_data = [self.iterations, self.evaluations,
                                  self.p_advantages, self.m_advantages,
                                  self.p_dist_sup, self.p_dist_mean,
                                  self.m_dist_sup, self.m_dist_mean,
                                  self.alfas, self.betas, self.p_change,
                                  self.m_change, self.bound,
                                  current[:, 0], current[:, 1],
                                  target[:, 0], target[:, 1]]

            if len(self.model_chooser.model_set) == 4:
                header_string = header_string + ';w_current[0];w_current[1];w_current[2];w_current[3]' \
                                                ';w_target[0];w_target[1];w_target[2];w_target[3]'

                current = np.array(self.w_current)
                target = np.array(self.w_target)

                execution_data = [self.iterations, self.evaluations,
                                  self.p_advantages, self.m_advantages,
                                  self.p_dist_sup, self.p_dist_mean,
                                  self.m_dist_sup, self.m_dist_mean,
                                  self.alfas, self.betas, self.p_change,
                                  self.m_change, self.bound,
                                  current[:, 0], current[:, 1],
                                  current[:, 2], current[:, 3],
                                  target[:, 0], target[:, 1],
                                  target[:, 2], target[:, 3]]

        execution_data = np.array(execution_data).T
        
        if entries is not None:
            filter_indices = np.linspace(0, len(execution_data)-1, entries, dtype=int)
            execution_data = execution_data[filter_indices]
        
        import os
        os.makedirs(dir_path, exist_ok=True)
        np.savetxt(dir_path + '/' + file_name, execution_data,
                delimiter=';', header=header_string, fmt='%.30e')
        
        if self.verbose >= 1:
            print(f"✓ Results saved to: {dir_path}/{file_name}")