import numpy as np
from time import time
from utils import evaluator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from utils.tabular import TabularModel
from utils.tabular_factory import model_from_matrix

from utils.tabular_operations import model_mean_tv_distance, model_sup_tv_distance, model_equiv_check, model_convex_combination_set


class ModelChooser(object):
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA

    def choose(self, model, delta_mu_P, U):
        pass


class GreedyModelChooser(ModelChooser):
    def choose(self, model, delta_mu, U):
        # GREEDY POLICY COMPUTATION
        target_model_rep = self.greedy_model(U)
        # instantiation of a target policy object
        target_model = TabularModel(target_model_rep, self.nS, self.nA)

        # EXPECTED RELATIVE ADVANTAGE COMPUTATION
        er_advantage = evaluator.compute_model_er_advantage(target_model, model, U, delta_mu)

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = model_sup_tv_distance(target_model, model)
        distance_mean = model_mean_tv_distance(target_model, model, delta_mu)

        return er_advantage, distance_sup, distance_mean, target_model

    def greedy_model(self, U, tol=0.0):
        greedy_model_rep = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        # loop to give maximum probability to the greedy action,
        # if more than one is greedy then uniform on the greedy actions
        for s in range(self.nS):
            for a in range(self.nA):
                sa = s * self.nA + a
                u_array = U[sa]
                probabilities = np.zeros(self.nS)

                # uniform if more than one greedy
                max = np.max(u_array)
                s1 = np.argwhere(np.abs(u_array - max) <= tol).flatten()
                probabilities[s1] = 1. / len(s1)
                greedy_model_rep[s][a] = zip(probabilities, range(self.nS))

        return greedy_model_rep

class GPModelChooser(ModelChooser):
    def __init__(self, model_set, nS, nA, init_model_vector, beta, original_model, max_history=500, gp_update_frequency=30):
        super(GPModelChooser, self).__init__(nS, nA)
        kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 10)) + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")

        self.model_set = model_set
        self.n_models = len(self.model_set)
        self.prev_target_model_vector = init_model_vector
        self.original_model = original_model
        self.gp_update_frequency = gp_update_frequency  # only do fitting every n iterations
        self.iteration = 0
        self.max_history = max_history
        # GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            normalize_y=True,
            optimizer=None,
            alpha=1e-6
        )
        self.beta = beta
        # store explored points in parameter space
        self.experience_X = [list(np.eye(self.n_models)[i]) for i in range(self.n_models)]
        self.experience_y = []
        self.fitting_times = []
        self.prediction_times = []

    def random_simplex_points(self, n_points=1000):
        """
        Generate random points uniformly on the simplex
        Uses the "break-the-stick" method
        """
        # Generate exponential random variables
        exp_samples = np.random.exponential(1, size=(n_points, self.n_models))
        
        # Normalize to get uniform simplex samples
        return exp_samples / exp_samples.sum(axis=1, keepdims=True)

    def choose(self, model, delta_mu, U):
        if self.iteration == 0:
            # initial evaluations at the corners of the simplex
            for i in range(self.n_models):
                target_model = self.model_set[i]
                er_advantage = evaluator.compute_model_er_advantage(target_model, model, U, delta_mu)
                self.experience_y.append(er_advantage)

        self.iteration += 1
        if self.iteration % self.gp_update_frequency == 0:

            # fit GP to expected relative advantages
            # before fitting, recompute experience_y by re-evaluating each stored simplex point with the current (model, delta_mu, U)
            # with max_history cap to keep the fit time bounded
            refreshed_y = []
            for x in self.experience_X:
                target_model = model_convex_combination_set(self.original_model, self.model_set, model, x)
                refreshed_y.append(evaluator.compute_model_er_advantage(target_model, model, U, delta_mu))
            self.experience_y = refreshed_y

            start_time = time()
            self.gp.fit(self.experience_X, self.experience_y)
            fit_time = time() - start_time
            self.fitting_times.append(fit_time)


            # candidate points generation
            candidate_points = self.random_simplex_points(n_points=1000)
            # predict UCB for candidate points
            start_time = time()
            self.gp.predict(candidate_points, return_std=True)
            means, stds = self.gp.predict(candidate_points, return_std=True)
            # time-variant beta (optimism term) for growing exploration
            beta_t = self.beta * np.sqrt(2 * np.log(self.iteration + 1))
            ucb_values = means + beta_t * stds
            predict_time = time() - start_time
            self.prediction_times.append(predict_time)

            # candidate/target model sampling via acquisition function (GP-UCB)
            max_index = np.argmax(ucb_values)
            target_model_vector = candidate_points[max_index].tolist()
            self.prev_target_model_vector = target_model_vector

            # build target model from the selected point
            target_model = model_convex_combination_set(self.original_model, self.model_set, model, target_model_vector)

            self.experience_X.append(target_model_vector)
            er_advantage = evaluator.compute_model_er_advantage(target_model, model, U, delta_mu)
            self.experience_y.append(er_advantage)
            if len(self.experience_X) > self.max_history:
                self.experience_X = self.experience_X[-self.max_history:]
                self.experience_y = self.experience_y[-self.max_history:]
            
            print(f"---- GPModelChooser:\tFitted GP at iteration {self.iteration},\tselected target model vector: {target_model_vector},\targmax vector: {np.argmax(target_model_vector)},\tER advantage: {er_advantage:.4f}")
        
        else:
            # build target model from the selected point
            target_model = model_convex_combination_set(self.original_model, self.model_set, model, self.prev_target_model_vector)
            er_advantage = evaluator.compute_model_er_advantage(target_model, model, U, delta_mu)


        # POLICY DISTANCE COMPUTATIONS
        distance_sup = model_sup_tv_distance(target_model, model)
        distance_mean = model_mean_tv_distance(target_model, model, delta_mu)

        return er_advantage, distance_sup, distance_mean, target_model
    
    def save_gp_times(self, filepath_prefix):
        np.savetxt(f"{filepath_prefix}_gp_fitting_times.csv", np.array(self.fitting_times), delimiter=";")
        np.savetxt(f"{filepath_prefix}_gp_prediction_times.csv", np.array(self.prediction_times), delimiter=";")

class SetModelChooser(ModelChooser):
    def __init__(self, model_set, nS, nA):
        self.model_set = model_set
        self.n_models = len(self.model_set)
        super(SetModelChooser, self).__init__(nS, nA)
        self.iteration = 0

    def choose(self, model, delta_mu, U):
        er_advantages = np.zeros(self.n_models)

        for i in range(self.n_models):
            er_advantages[i] = evaluator.compute_model_er_advantage(self.model_set[i], model, U, delta_mu)

        index = np.argmax(er_advantages)
        target_model = self.model_set[index]
        er_advantage = er_advantages[index]

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = model_sup_tv_distance(target_model, model)
        distance_mean = model_mean_tv_distance(target_model, model, delta_mu)

        self.iteration += 1
        if self.iteration % 100 == 0:
            print(f"---- SetModelChooser:\tIteration {self.iteration},\tselected target model index: {index},\tER advantage: {er_advantage:.4f}")

        return er_advantage, distance_sup, distance_mean, target_model

    def set(self, model, delta_mu, U):
        er_advantages = np.zeros(self.n_models)
        sup_distances = np.zeros(self.n_models)
        mean_distances = np.zeros(self.n_models)

        # advantages and distances computations
        for i in range(self.n_models):
            er_advantages[i] = evaluator.compute_model_er_advantage(self.model_set[i], model, U, delta_mu)
            sup_distances[i] = model_sup_tv_distance(self.model_set[i], model)
            mean_distances[i] = model_mean_tv_distance(self.model_set[i], model, delta_mu)

        return er_advantages, sup_distances, mean_distances


class DoNotCreateTransitionsGreedyModelChooser(ModelChooser):
    def __init__(self, original_model, nS, nA):
        self.original_model = original_model
        super(DoNotCreateTransitionsGreedyModelChooser, self).__init__(nS, nA)

    def choose(self, model, delta_mu, U):
        target_model_rep = self.dnct_greedy_model(U)
        target_model = TabularModel(target_model_rep, self.nS, self.nA)

        er_advantage = evaluator.compute_model_er_advantage(target_model, model, U, delta_mu)

        distance_sup = model_sup_tv_distance(target_model, model)
        distance_mean = model_mean_tv_distance(target_model, model,
                                                 delta_mu)

        return er_advantage, distance_sup, distance_mean, target_model

    def dnct_greedy_model(self, U, tol=0.0):
        greedy_model_rep = {s: {a: [] for a in range(self.nA)} for s in
                            range(self.nS)}

        for s in range(self.nS):
            for a in range(self.nA):
                sa = s * self.nA + a
                u_array = np.copy(U[sa])

                li = self.original_model[s][a]
                for elem in li:
                    if elem[0] == 0.:
                        u_array[elem[1]] = -np.inf

                probabilities = np.zeros(self.nS)

                # uniform if more than one greedy
                max = np.max(u_array)

                s1 = np.argwhere(np.abs(u_array - max) <= tol).flatten()
                probabilities[s1] = 1. / len(s1)
                greedy_model_rep[s][a] = zip(probabilities, range(self.nS))

        return greedy_model_rep
