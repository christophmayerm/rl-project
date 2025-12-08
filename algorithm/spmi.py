
from algorithm.model_chooser import *
from utils.evaluator import *
from utils.tabular import *
from algorithm.logger import Logger
from utils import evaluator

from algorithm.policy_chooser import *
from utils.tabular_operations import policy_convex_combination, model_convex_combination, policy_equiv_check, model_equiv_check
    

class SPMI(object):

    def __init__(self,
        conf_mdp,
        eps,
        policy_chooser=None,
        model_chooser=None,
        max_iter=10000,
        delta_q=None,
        persistent=True,
        # NEW PARAMETERS FOR SA-PMI
        adversarial_model_chooser=None,
        curriculum_schedule='linear',
        B_min=0.01,
        B_max=0.3,
        K_warmup=50,
        metrics_evaluator=None):  # NEW PARAMETER

        '''
        This class allows to instantiate a Safe Policy Model Iterator object, i.e., an object exposing
        the methods to perform policy-model learning on a given Conf-MDP. This class implements the
        Safe Policy-Model Iteration along with its derivatives.

        :param conf_mdp: the Configurable Markov Decision Process object
        :param eps: threshold to be use to stop iterations
        :param policy_chooser: an object implementing the choice of the target policy
        :param model_chooser: an object implementing the choice of the target model
        :param max_iter: maximum number of iterations to be performed
        :param delta_q: the value of DeltaQ, if None 1/(1-gamma) is used
        :param persistent: whether to adopt the persistent target selection instead of the simple greedy
        ADDITIONAL PARAMETERS FOR SA-PMI:
        :param adversarial_model_chooser: chooser for adversarial model selection
        :param curriculum_schedule: 'linear', 'exponential', or 'step'
        :param B_min: minimum adversarial budget (initial curriculum difficulty)
        :param B_max: maximum adversarial budget (final curriculum difficulty)
        :param K_warmup: number of iterations to reach full adversarial strength
        '''

        # --------------------------------------
        # ----- ATTRIBUTES INITIALIZATIONS -----
        # --------------------------------------
        self.mdp = conf_mdp
        self.gamma = conf_mdp.gamma
        self.horizon = conf_mdp.horizon
        self.eps = eps
        self.iteration_horizon = max_iter
        self.persistent = persistent
        # default delta_q = (1-gamma^H)/(1-gamma)
        if delta_q is None:
            self.delta_q = (1. - self.gamma ** self.horizon) / (1 - self.gamma)
        else:
            self.delta_q = delta_q
        # default policy_chooser instantiation if none
        if policy_chooser is None:
            self.policy_chooser = GreedyPolicyChooser(conf_mdp.nS, conf_mdp.nA)
        else:
            self.policy_chooser = policy_chooser
        # default model_chooser instantiation if none
        if model_chooser is None:
            self.model_chooser = GreedyModelChooser(conf_mdp.nS, conf_mdp.nA)
        else:
            self.model_chooser = model_chooser

        # LOGGER INSTANTIATION
        self.logger = Logger(self.mdp, self.model_chooser, metrics_evaluator=metrics_evaluator)
        # NEW: Adversarial components (add at the end of __init__)
        if adversarial_model_chooser is None:
            from algorithm.adversarial_model_chooser import GreedyAdversarialChooser
            self.adversarial_model_chooser = GreedyAdversarialChooser(
                conf_mdp.nS, conf_mdp.nA,
                budget_schedule=curriculum_schedule,
                B_min=B_min, B_max=B_max, K_warmup=K_warmup
            )
        else:
            self.adversarial_model_chooser = adversarial_model_chooser
        
        # Store nominal model reference
        self.nominal_model = None

    # -------------------------------------
    # ----- ALGORITHMS IMPLEMENTATION -----
    # -------------------------------------

    # implementation of Safe Policy-Model Iteration
    def spmi(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.env.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        if self.logger.iteration == 0:
            reward_matrix = reward.get_matrix()
            nonzero_count = np.sum(reward_matrix != 0)
            print(f"\n🔍 Reward Matrix Check:")
            print(f"   Shape: {reward_matrix.shape}")
            print(f"   Non-zero rewards: {nonzero_count}")
            if nonzero_count == 0:
                print(f"   ⚠️  ERROR: All rewards are zero!")
            else:
                print(f"   ✓ Rewards look good")
                max_reward = reward_matrix.max()
                print(f"   Max reward: {max_reward:.6f}\n")
        policy = initial_policy
        model = initial_model

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY-MODEL UPDATE LOOP
        # the policy and the model are continuously updated until the iteration_horizon is reached or
        # the relative advantages fall below the convergence threshold
        while (p_er_adv > convergence or m_er_adv > convergence) and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # loop to select the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24)
                    alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24) - .5 * \
                                (m_dist_mean / (p_dist_mean + 1e-24) + m_dist_sup / (p_dist_sup + 1e-24))
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_mean)
                    beta1 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_mean) - .5 / gamma * \
                                (p_dist_mean / (m_dist_mean + 1e-24) + p_dist_sup / (m_dist_sup + 1e-24))

                    alpha0 = np.clip(alpha0, 0., 1.)
                    alpha1 = np.clip(alpha1, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)
                    beta1 = np.clip(beta1, 0., 1.)

                    for alpha, beta in [(alpha0, 0.), (0., beta0), (alpha1, 1.), (1., beta1)]:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean +
                                 gamma * (beta ** 2) * m_dist_sup * m_dist_mean +
                                 alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                convergence, bound_star)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model


    # implementation of SPMI-sup,
    # version of SPMI adopting a looser lower bound on performance improvement
    # in which the mean distances are replaced by sup distances
    def spmi_sup(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.env.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY-MODEL UPDATE LOOP
        # the policy and the model are continuously updated until the iteration_horizon is reached or
        # the relative advantages fall below the convergence threshold
        while ((p_er_adv + m_er_adv) > convergence) and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # loop to select the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_sup + 1e-24)
                    alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_sup + 1e-24) - \
                                m_dist_sup / (p_dist_sup + 1e-24)
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_sup)
                    beta1 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_sup) - 1. / gamma * \
                                p_dist_sup / (m_dist_sup + 1e-24)

                    alpha0 = np.clip(alpha0, 0., 1.)
                    alpha1 = np.clip(alpha1, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)
                    beta1 = np.clip(beta1, 0., 1.)

                    for alpha, beta in [(alpha0, 0.), (0., beta0), (alpha1, 1.), (1., beta1)]:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup ** 2 +
                                 gamma * (beta ** 2) * m_dist_sup ** 2 +
                                 2 * alpha * beta * p_dist_sup * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                convergence, bound_star)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model


    # implementation of SPMI-alt,
    # version of SPMI executing alternated improvement of the policy and the model
    # regardless of the value of the bound
    def spmi_alt(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.env.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(
            model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY-MODEL UPDATE LOOP
        # the policy and the model are continuously updated until the iteration_horizon is reached or
        # the relative advantages fall below the convergence threshold
        while (p_er_adv > convergence or m_er_adv > convergence) and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = -1.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # loop selecting the update of the policy and the model alternately
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24)
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_mean + 1e-24)

                    alpha0 = np.clip(alpha0, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)

                    if self.logger.iteration % 2 == 0:
                        li = [(alpha0, 0.)]
                    else:
                        li = [(0., beta0)]

                    for alpha, beta in li:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean +
                                 gamma * (beta ** 2) * m_dist_sup * m_dist_mean +
                                 alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                convergence, bound_star)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model


    # implementation of SPMI-nofull,
    # version of SPMI that prevents the selection of the "full step" updates,
    # i.e., the update (alfa_star,1) and (1,beta_star).
    # At each iteration, only one between the policy and the model is updated
    def spmi_no_full(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.env.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(
            model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY-MODEL UPDATE LOOP
        # the policy and the model are continuously updated until the iteration_horizon is reached or
        # the relative advantages fall below the convergence threshold
        while ((p_er_adv + m_er_adv) > convergence) and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # loop to select the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24)
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_mean)

                    alpha0 = np.clip(alpha0, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)

                    for alpha, beta in [(alpha0, 0.), (0., beta0)]:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean +
                                 gamma * (beta ** 2) * m_dist_sup * m_dist_mean +
                                 alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                convergence, bound_star)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model


    # implementation of SPI+SMI,
    # performing a complete execution of Safe Policy Iteration (SPI)
    # followed by a complete execution of Safe Model Iteration (SMI)
    def spi_smi(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.env.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # ------ SPI ------

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY UPDATE LOOP
        # the policy is continuously updated until the iteration_horizon is reached or
        # the relative advantage falls below the convergence threshold
        while p_er_adv > convergence and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            target_policy_star = target_policy
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None

            # selection of the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:

                alpha = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                            p_dist_sup * p_dist_mean + 1e-24)

                alpha = np.clip(alpha, 0., 1.)

                bound = alpha * p_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        ((alpha ** 2) * p_dist_sup * p_dist_mean)

                if bound > bound_star:
                    bound_star = bound
                    alpha_star = alpha
                    target_policy_star = target_policy
                    p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean

            # policy update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, alpha_star, np.nan, p_er_adv_star, np.nan,
                       p_dist_sup_star, p_dist_mean_star, np.nan, np.nan,
                       target_policy_star, target_policy_old, np.nan,
                       np.nan, convergence, bound_star)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

        # ------ SMI ------

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # MODEL UPDATE LOOP
        # the model is continuously updated until the iteration_horizon is reached or
        # the relative advantage falls below the convergence threshold
        while m_er_adv > convergence and self.logger.iteration < iteration_horizon:

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            beta_star = 0.
            target_model_star = target_model
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # selection of the update yielding the maximum bound value
            for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                beta = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                            * m_dist_sup * m_dist_mean)

                beta = np.clip(beta, 0., 1.)

                bound = beta * m_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                         gamma * (beta ** 2) * m_dist_sup * m_dist_mean

                if bound > bound_star:
                    bound_star = bound
                    beta_star = beta
                    target_model_star = target_model
                    m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # model update
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, np.nan, beta_star, np.nan, m_er_adv_star,
                       np.nan, np.nan, m_dist_sup_star, m_dist_mean_star,
                       np.nan, np.nan, target_model_star,
                       target_model_old, convergence, bound_star)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model

    # implementation of SMI+SPI,
    # performing a complete execution of Safe Model Iteration (SMI)
    # followed by a complete execution of Safe Policy Iteration (SPI)
    def smi_spi(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.env.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # ------ SMI ------

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # MODEL UPDATE LOOP
        # the model is continuously updated until the iteration_horizon is reached or
        # the relative advantage falls below the convergence threshold
        while m_er_adv > convergence and self.logger.iteration < iteration_horizon:

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model,
                                                                                                             target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            beta_star = 0.
            target_model_star = target_model
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # selection of the update yielding the maximum bound value
            for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                beta = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                                   * m_dist_sup * m_dist_mean)

                beta = np.clip(beta, 0., 1.)

                bound = beta * m_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        gamma * (beta ** 2) * m_dist_sup * m_dist_mean

                if bound > bound_star:
                    bound_star = bound
                    beta_star = beta
                    target_model_star = target_model
                    m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # model update
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, np.nan, beta_star, np.nan, m_er_adv_star,
                       np.nan, np.nan, m_dist_sup_star, m_dist_mean_star,
                       np.nan, np.nan, target_model_star,
                       target_model_old, convergence, bound_star)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        # ------ SPI ------

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY UPDATE LOOP
        # the policy is continuously updated until the iteration_horizon is reached or
        # the relative advantage falls below the convergence threshold
        while p_er_adv > convergence and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy,
                                                                                                            target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            target_policy_star = target_policy
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None

            # selection of the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:

                alpha = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                                    p_dist_sup * p_dist_mean + 1e-24)

                alpha = np.clip(alpha, 0., 1.)

                bound = alpha * p_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        ((alpha ** 2) * p_dist_sup * p_dist_mean)

                if bound > bound_star:
                    bound_star = bound
                    alpha_star = alpha
                    target_policy_star = target_policy
                    p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean

            # policy update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, alpha_star, np.nan, p_er_adv_star, np.nan,
                       p_dist_sup_star, p_dist_mean_star, np.nan, np.nan,
                       target_policy_star, target_policy_old, np.nan,
                       np.nan, convergence, bound_star)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

        return policy, model

    def sa_pmi(self, initial_policy, initial_model, nominal_model=None):
        """
        Safe Adversarial Policy-Model Iteration with Curriculum Learning
        
        Extends SPMI with adversarial model perturbations that grow according
        to a curriculum schedule, creating robust policies against worst-case
        environment configurations within a bounded budget.
        
        Algorithm:
        1. Inner loop: Adversary selects worst-case model M_a within budget B(k)
        2. Outer loop: Agent optimizes policy π and cooperative model M_c
        3. Budget B(k) grows over time (curriculum learning)
        
        :param initial_policy: starting policy π_0
        :param initial_model: starting cooperative model M_c^0
        :param nominal_model: baseline model for adversarial reference (if None, uses initial_model)
        :return: (robust_policy, cooperative_model, final_adversarial_model)
        """
        
        # ==========================================
        # INITIALIZATION
        # ==========================================
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        reward = TabularReward(self.mdp.env.P, self.mdp.nS, self.mdp.nA)
        self.logger.reset()
        
        policy = initial_policy
        model_coop = initial_model  # Cooperative model M_c
        
        # Initialize nominal model for adversarial reference
        if nominal_model is None:
            nominal_model = initial_model
        self.nominal_model = nominal_model
        
        # Initialize adversarial model M_a = M_c (no perturbation initially)
        model_adv = initial_model
        
        # ==========================================
        # INITIAL TARGET SELECTION
        # ==========================================
        # Compute initial Q and choose target policy
        Q = evaluator.compute_q_function(policy, model_adv, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model_adv, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy
        
        # Compute initial U and choose target cooperative model
        U = evaluator.compute_u_function(policy, model_adv, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model_adv, gamma, horizon, nS, nA, d_mu)
        mc_er_adv, mc_dist_sup, mc_dist_mean, target_model_coop = self.model_chooser.choose(model_coop, delta_mu, U)
        target_model_coop_old = target_model_coop
        
        # Convergence threshold
        convergence = eps / (1 - gamma)
        
        # ==========================================
        # MAIN ADVERSARIAL POLICY-MODEL UPDATE LOOP
        # ==========================================
        while (p_er_adv > convergence or mc_er_adv > convergence) and self.logger.iteration < iteration_horizon:
            
            # ==========================================
            # STEP 1: ADVERSARIAL MODEL UPDATE (Inner Loop)
            # ==========================================
            # Adversary chooses worst-case model M_a within curriculum budget B(k)
            model_adv, ma_er_disadvantage, ma_dist_sup, ma_dist_mean, current_budget = \
                self.adversarial_model_chooser.choose_adversarial(
                    model_coop, nominal_model, delta_mu, U, iteration=self.logger.iteration
                )
            
            # Evaluate under adversarial model (worst-case scenario)
            Q_adv = evaluator.compute_q_function(policy, model_adv, reward, gamma, horizon=horizon)
            U_adv = evaluator.compute_u_function(policy, model_adv, reward, gamma, horizon=horizon)
            d_mu_adv = evaluator.compute_discounted_s_distribution(mu, policy, model_adv, gamma, horizon, nS, nA)
            delta_mu_adv = evaluator.compute_discounted_sa_distribution(mu, policy, model_adv, gamma, horizon, nS, nA, d_mu_adv)
            
            # ==========================================
            # STEP 2: BUILD TARGET LISTS (Persistent Choice)
            # ==========================================
            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q_adv, d_mu_adv)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu_adv)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))
            
            target_models_coop = [(target_model_coop, mc_er_adv, mc_dist_sup, mc_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model_coop, target_model_coop_old) and not model_equiv_check(model_coop, target_model_coop_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_coop_old, model_coop, U_adv, delta_mu_adv)
                    dist_sup_old = policy_sup_tv_distance(target_model_coop_old, model_coop)
                    dist_mean_old = policy_mean_tv_distance(target_model_coop_old, model_coop, delta_mu_adv)
                    target_models_coop.append((target_model_coop_old, er_adv_old, dist_sup_old, dist_mean_old))
            
            # ==========================================
            # STEP 3: OPTIMIZE ROBUST BOUND
            # ==========================================
            bound_star = -np.inf  # Allow negative bounds initially (adversarial setting)
            alpha_star = 0.
            beta_coop_star = 0.
            target_policy_star = target_policy
            target_model_coop_star = target_model_coop
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            mc_er_adv_star, mc_dist_sup_star, mc_dist_mean_star = None, None, None
            
            # Search for optimal (α, β_c) under adversarial model
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model_coop, mc_er_adv, mc_dist_sup, mc_dist_mean in target_models_coop:
                    
                    # Compute update coefficients with adversarial penalty
                    # These formulas account for the adversarial perturbation
                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * 
                                p_dist_sup * p_dist_mean + 1e-24)
                    alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24) - .5 * \
                                ((mc_dist_mean + ma_dist_mean) / (p_dist_mean + 1e-24) + 
                                (mc_dist_sup + ma_dist_sup) / (p_dist_sup + 1e-24))
                    
                    beta_c0 = ((1 - gamma) * mc_er_adv) / (self.delta_q * (gamma ** 2)
                                * mc_dist_sup * mc_dist_mean + 1e-24)
                    beta_c1 = ((1 - gamma) * mc_er_adv) / (self.delta_q * (gamma ** 2)
                                * mc_dist_sup * mc_dist_mean + 1e-24) - .5 / gamma * \
                                ((p_dist_mean + ma_dist_mean) / (mc_dist_mean + 1e-24) + 
                                (p_dist_sup + ma_dist_sup) / (mc_dist_sup + 1e-24))
                    
                    alpha0 = np.clip(alpha0, 0., 1.)
                    alpha1 = np.clip(alpha1, 0., 1.)
                    beta_c0 = np.clip(beta_c0, 0., 1.)
                    beta_c1 = np.clip(beta_c1, 0., 1.)
                    
                    # Evaluate bound for each configuration
                    for alpha, beta_c in [(alpha0, 0.), (0., beta_c0), (alpha1, 1.), (1., beta_c1)]:
                        
                        # ROBUST BOUND: includes adversarial penalty
                        # This ensures monotonic improvement under worst-case adversarial model
                        adversarial_penalty = (gamma / (1 - gamma) * self.delta_q / 2) * \
                                            (ma_dist_sup * ma_dist_mean)       
                        
                        bound = alpha * p_er_adv + beta_c * mc_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean +
                                gamma * (beta_c ** 2) * mc_dist_sup * mc_dist_mean +
                                alpha * beta_c * p_dist_sup * mc_dist_mean + 
                                alpha * beta_c * p_dist_mean * mc_dist_sup) - \
                                adversarial_penalty
                        
                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_coop_star = beta_c
                            target_policy_star = target_policy
                            target_model_coop_star = target_model_coop
                            p_er_adv_star = p_er_adv
                            p_dist_sup_star, p_dist_mean_star = p_dist_sup, p_dist_mean
                            mc_er_adv_star = mc_er_adv
                            mc_dist_sup_star, mc_dist_mean_star = mc_dist_sup, mc_dist_mean
            
            # ==========================================
            # STEP 4: POLICY AND COOPERATIVE MODEL UPDATE
            # ==========================================
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_coop_star > 0:
                model_coop = self.model_combination(beta_coop_star, target_model_coop_star, model_coop)
            
            # ==========================================
            # STEP 5: PERFORMANCE EVALUATION (under adversarial model)
            # ==========================================
            Q = evaluator.compute_q_function(policy, model_adv, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model_adv, gamma, horizon, nS, nA)
            
            # Log with adversarial metrics
            # With:
            self.logger.update(
                J_p_m, alpha_star, beta_coop_star, 
                p_er_adv_star, mc_er_adv_star,
                p_dist_sup_star, p_dist_mean_star, 
                mc_dist_sup_star, mc_dist_mean_star,
                target_policy_star, target_policy_old, 
                target_model_coop_star, target_model_coop_old,
                convergence, bound_star,
                adversarial_budget=current_budget,
                adversarial_disadvantage=ma_er_disadvantage,
                adversarial_dist_sup=ma_dist_sup,
                adversarial_dist_mean=ma_dist_mean,
                # NEW: Pass policy, model, and distributions for metrics
                policy=policy,
                model=model_adv,  # Evaluate under adversarial model
                reward=reward,
                d_mu=d_mu,
                delta_mu=delta_mu
            )
            
            # ==========================================
            # STEP 6: SELECT NEXT TARGETS (under adversarial model for robustness)
            # ==========================================
            # Choose next target policy (evaluate under adversarial model)
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model_adv, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
            
            # Choose next target cooperative model
            target_model_coop_old = target_model_coop_star
            U = evaluator.compute_u_function(policy, model_adv, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model_adv, gamma, horizon, nS, nA, d_mu)
            mc_er_adv, mc_dist_sup, mc_dist_mean, target_model_coop = self.model_chooser.choose(model_coop, delta_mu, U)
        
        return policy, model_coop, model_adv
    
    """
    SA-PMI v2: Adaptive Curriculum Integration
    Add this method to your SPMI class in algorithm/spmi.py
    """

    def sa_pmi_v2(self, initial_policy, initial_model, nominal_model=None):
        """
        Safe Adversarial Policy-Model Iteration v2 with Adaptive Curriculum
        
        Key improvements over sa_pmi:
        1. Adaptive budget based on learning progress and stability
        2. Multi-phase curriculum (warmup → gradual → robust)
        3. Safety constraints based on TV distance monitoring
        4. Emergency mode for model divergence
        5. Smooth budget transitions to prevent oscillation
        
        Args:
            initial_policy: Starting policy π_0
            initial_model: Starting cooperative model M_c^0
            nominal_model: Baseline model for adversarial reference (if None, uses initial_model)
            
        Returns:
            (robust_policy, cooperative_model, final_adversarial_model)
        """
        
        # ==========================================
        # INITIALIZATION
        # ==========================================
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        reward = TabularReward(self.mdp.env.P, self.mdp.nS, self.mdp.nA)
        self.logger.reset()
        
        policy = initial_policy
        model_coop = initial_model  # Cooperative model M_c
        
        # Initialize nominal model for adversarial reference
        if nominal_model is None:
            nominal_model = copy.deepcopy(initial_model)
        self.nominal_model = nominal_model
        
        # Initialize adversarial model M_a = M_c (no perturbation initially)
        model_adv = copy.deepcopy(initial_model)
        
        # ==========================================
        # ADAPTIVE CURRICULUM SETUP
        # ==========================================
        from adaptive_curriculum import AdaptiveCurriculumScheduler
        from adaptive_adversarial_chooser import AdaptiveAdversarialChooser
        
        # Create adaptive curriculum scheduler
        curriculum = AdaptiveCurriculumScheduler(
            B_max=self.adversarial_model_chooser.B_max,
            B_min=self.adversarial_model_chooser.B_min,
            # Use parameters from initialization or defaults
            performance_window=100,
            convergence_trigger=eps * 10,  # Start adversarial when improvement < 10*eps
            tv_safe_threshold=0.65,
            tv_danger_threshold=0.75,
            stability_threshold=0.05,
            warmup_phase_ratio=0.4,
            gradual_phase_ratio=0.4,
            robust_phase_ratio=0.2,
            min_iterations_before_adversarial=500,
        )
        
        # Create adaptive adversarial chooser
        adaptive_adversary = AdaptiveAdversarialChooser(
            nS=nS,
            nA=nA,
            curriculum_scheduler=curriculum
        )
        
        print(f"\n{'='*70}")
        print(f"🎯 SA-PMI v2: Adaptive Curriculum Learning")
        print(f"{'='*70}")
        print(f"Budget range: [{curriculum.B_min:.4f}, {curriculum.B_max:.4f}]")
        print(f"Safety thresholds: TV_safe={curriculum.tv_safe_threshold}, TV_danger={curriculum.tv_danger_threshold}")
        print(f"Convergence trigger: {curriculum.convergence_trigger:.6f}")
        print(f"Multi-phase curriculum: warmup({curriculum.warmup_phase_ratio:.0%}) → "
            f"gradual({curriculum.gradual_phase_ratio:.0%}) → robust({curriculum.robust_phase_ratio:.0%})")
        print(f"{'='*70}\n")
        
        # ==========================================
        # INITIAL TARGET SELECTION
        # ==========================================
        # Compute initial Q and choose target policy
        Q = evaluator.compute_q_function(policy, model_adv, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model_adv, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy
        
        # Compute initial U and choose target cooperative model
        U = evaluator.compute_u_function(policy, model_adv, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model_adv, gamma, horizon, nS, nA, d_mu)
        mc_er_adv, mc_dist_sup, mc_dist_mean, target_model_coop = self.model_chooser.choose(model_coop, delta_mu, U)
        target_model_coop_old = target_model_coop
        
        # Convergence threshold
        convergence = eps / (1 - gamma)
        
        # ==========================================
        # MAIN ADAPTIVE POLICY-MODEL UPDATE LOOP
        # ==========================================
        print(f"Starting training with {iteration_horizon} max iterations...\n")
        
        while (p_er_adv > convergence or mc_er_adv > convergence) and self.logger.iteration < iteration_horizon:
            
            # ==========================================
            # STEP 1: COMPUTE ADAPTIVE BUDGET
            # ==========================================
            # First, compute current model divergence for budget adaptation
            if self.logger.iteration > 0:
                current_tv_mean = policy_mean_tv_distance(model_adv, nominal_model, delta_mu)
                current_tv_max = policy_sup_tv_distance(model_adv, nominal_model)
            else:
                current_tv_mean = 0.0
                current_tv_max = 0.0
            
            # Get adaptive budget from curriculum scheduler
            current_budget, curriculum_state = curriculum.get_budget(
                iteration=self.logger.iteration,
                max_iterations=iteration_horizon,
                logger=self.logger,
                model_tv_mean=current_tv_mean,
                model_tv_max=current_tv_max
            )
            
            # Print curriculum state periodically
            if self.logger.iteration % 100 == 0 and curriculum.adversarial_active:
                print(f"\n[Iter {self.logger.iteration}] Curriculum State:")
                print(f"  Phase: {curriculum_state.phase}")
                print(f"  Budget: {current_budget:.4f} (base: {curriculum_state.base_budget:.4f}, "
                    f"safety: {curriculum_state.safety_scale:.2f})")
                print(f"  Stability: {curriculum_state.stability_score:.3f}, "
                    f"Trend: {curriculum_state.performance_trend:.4f}")
                print(f"  TV distance: {current_tv_mean:.3f}")
                print(f"  Reason: {curriculum_state.reason}\n")
            
            # ==========================================
            # STEP 2: ADVERSARIAL MODEL UPDATE
            # ==========================================
            # Adversary chooses worst-case model M_a within adaptive budget
            model_adv, ma_er_disadvantage, ma_dist_sup, ma_dist_mean, budget_used = \
                adaptive_adversary.choose_adversarial(
                    model_coop=model_coop,
                    nominal_model=nominal_model,
                    delta_mu=delta_mu,
                    U=U,
                    iteration=self.logger.iteration,
                    logger=self.logger,
                    override_budget=current_budget
                )
            
            # Evaluate under adversarial model (worst-case scenario)
            Q_adv = evaluator.compute_q_function(policy, model_adv, reward, gamma, horizon=horizon)
            U_adv = evaluator.compute_u_function(policy, model_adv, reward, gamma, horizon=horizon)
            d_mu_adv = evaluator.compute_discounted_s_distribution(mu, policy, model_adv, gamma, horizon, nS, nA)
            delta_mu_adv = evaluator.compute_discounted_sa_distribution(mu, policy, model_adv, gamma, horizon, nS, nA, d_mu_adv)
            
            # ==========================================
            # STEP 3: BUILD TARGET LISTS (Persistent Choice)
            # ==========================================
            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q_adv, d_mu_adv)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu_adv)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))
            
            target_models_coop = [(target_model_coop, mc_er_adv, mc_dist_sup, mc_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model_coop, target_model_coop_old) and not model_equiv_check(model_coop, target_model_coop_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_coop_old, model_coop, U_adv, delta_mu_adv)
                    dist_sup_old = policy_sup_tv_distance(target_model_coop_old, model_coop)
                    dist_mean_old = policy_mean_tv_distance(target_model_coop_old, model_coop, delta_mu_adv)
                    target_models_coop.append((target_model_coop_old, er_adv_old, dist_sup_old, dist_mean_old))
            
            # ==========================================
            # STEP 4: OPTIMIZE ROBUST BOUND
            # ==========================================
            bound_star = -np.inf  # Allow negative bounds in adversarial setting
            alpha_star = 0.
            beta_coop_star = 0.
            target_policy_star = target_policy
            target_model_coop_star = target_model_coop
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            mc_er_adv_star, mc_dist_sup_star, mc_dist_mean_star = None, None, None
            
            # Search for optimal (α, β_c) under adversarial model
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model_coop, mc_er_adv, mc_dist_sup, mc_dist_mean in target_models_coop:
                    
                    # Compute update coefficients with adversarial penalty
                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * 
                                p_dist_sup * p_dist_mean + 1e-24)
                    alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24) - .5 * \
                                ((mc_dist_mean + ma_dist_mean) / (p_dist_mean + 1e-24) + 
                                (mc_dist_sup + ma_dist_sup) / (p_dist_sup + 1e-24))
                    
                    beta_c0 = ((1 - gamma) * mc_er_adv) / (self.delta_q * (gamma ** 2)
                                * mc_dist_sup * mc_dist_mean + 1e-24)
                    beta_c1 = ((1 - gamma) * mc_er_adv) / (self.delta_q * (gamma ** 2)
                                * mc_dist_sup * mc_dist_mean + 1e-24) - .5 / gamma * \
                                ((p_dist_mean + ma_dist_mean) / (mc_dist_mean + 1e-24) + 
                                (p_dist_sup + ma_dist_sup) / (mc_dist_sup + 1e-24))
                    
                    alpha0 = np.clip(alpha0, 0., 1.)
                    alpha1 = np.clip(alpha1, 0., 1.)
                    beta_c0 = np.clip(beta_c0, 0., 1.)
                    beta_c1 = np.clip(beta_c1, 0., 1.)
                    
                    # Evaluate bound for each configuration
                    for alpha, beta_c in [(alpha0, 0.), (0., beta_c0), (alpha1, 1.), (1., beta_c1)]:
                        
                        # ROBUST BOUND: includes adversarial penalty
                        adversarial_penalty = (gamma / (1 - gamma) * self.delta_q / 2) * \
                                            gamma * (ma_dist_sup * ma_dist_mean)
                        
                        bound = alpha * p_er_adv + beta_c * mc_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean +
                                gamma * (beta_c ** 2) * mc_dist_sup * mc_dist_mean +
                                alpha * beta_c * p_dist_sup * mc_dist_mean + 
                                alpha * beta_c * p_dist_mean * mc_dist_sup) - \
                                adversarial_penalty
                        
                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_coop_star = beta_c
                            target_policy_star = target_policy
                            target_model_coop_star = target_model_coop
                            p_er_adv_star = p_er_adv
                            p_dist_sup_star, p_dist_mean_star = p_dist_sup, p_dist_mean
                            mc_er_adv_star = mc_er_adv
                            mc_dist_sup_star, mc_dist_mean_star = mc_dist_sup, mc_dist_mean
            
            # ==========================================
            # STEP 5: POLICY AND COOPERATIVE MODEL UPDATE
            # ==========================================
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_coop_star > 0:
                model_coop = self.model_combination(beta_coop_star, target_model_coop_star, model_coop)
            
            # ==========================================
            # STEP 6: PERFORMANCE EVALUATION (under adversarial model)
            # ==========================================
            Q = evaluator.compute_q_function(policy, model_adv, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model_adv, gamma, horizon, nS, nA)
            
            # Log with curriculum metrics
            self.logger.update(
                J_p_m, alpha_star, beta_coop_star, 
                p_er_adv_star, mc_er_adv_star,
                p_dist_sup_star, p_dist_mean_star, 
                mc_dist_sup_star, mc_dist_mean_star,
                target_policy_star, target_policy_old, 
                target_model_coop_star, target_model_coop_old,
                convergence, bound_star,
                # Curriculum-specific metrics
                adversarial_budget=current_budget,
                adversarial_disadvantage=ma_er_disadvantage,
                adversarial_dist_sup=ma_dist_sup,
                adversarial_dist_mean=ma_dist_mean,
                curriculum_phase=curriculum_state.phase,
                curriculum_safety_scale=curriculum_state.safety_scale,
                curriculum_stability=curriculum_state.stability_score,
                # Pass for auxiliary metrics if evaluator present
                policy=policy,
                model=model_adv,
                reward=reward,
                d_mu=d_mu_adv,
                delta_mu=delta_mu_adv
            )
            
            # ==========================================
            # STEP 7: SELECT NEXT TARGETS (under adversarial model)
            # ==========================================
            # Choose next target policy (evaluate under adversarial model for robustness)
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model_adv, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
            
            # Choose next target cooperative model
            target_model_coop_old = target_model_coop_star
            U = evaluator.compute_u_function(policy, model_adv, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model_adv, gamma, horizon, nS, nA, d_mu)
            mc_er_adv, mc_dist_sup, mc_dist_mean, target_model_coop = self.model_chooser.choose(model_coop, delta_mu, U)
        
        # ==========================================
        # POST-TRAINING ANALYSIS
        # ==========================================
        print(f"\n{'='*70}")
        print(f"🎯 SA-PMI v2 Training Complete")
        print(f"{'='*70}\n")
        
        # Curriculum summary
        curriculum_summary = curriculum.get_curriculum_summary()
        print("📊 Curriculum Learning Summary:")
        print(f"  Total iterations: {curriculum_summary['total_iterations']}")
        print(f"  Adversarial start: iteration {curriculum_summary.get('adversarial_start', 'N/A')}")
        print(f"  Phase distribution: {curriculum_summary.get('phase_distribution', {})}")
        print(f"  Budget stats: mean={curriculum_summary['budget_stats']['mean']:.4f}, "
            f"max={curriculum_summary['budget_stats']['max']:.4f}")
        print(f"  Safety events: {curriculum_summary['safety_events']} "
            f"(emergency: {curriculum_summary['emergency_events']})")
        print(f"  Budget-performance correlation: {curriculum_summary['budget_performance_correlation']:.3f}")
        
        # Adversarial summary
        adv_summary = adaptive_adversary.get_adversarial_summary()
        print(f"\n🎭 Adversarial Training Summary:")
        print(f"  Adversarial iterations: {adv_summary.get('total_adversarial_iterations', 0)}")
        print(f"  Avg budget used: {adv_summary.get('avg_budget', 0):.4f}")
        print(f"  Avg disadvantage: {adv_summary.get('avg_disadvantage', 0):.4f}")
        print(f"  Avg TV distance: {adv_summary.get('avg_dist_mean', 0):.4f}")
        print(f"  Max TV distance: {adv_summary.get('max_dist_mean', 0):.4f}")
        
        print(f"\n{'='*70}\n")
        
        # Save curriculum history plot
        if hasattr(self.logger, 'output_dir') and self.logger.output_dir:
            import os
            os.makedirs(self.logger.output_dir, exist_ok=True)
            curriculum_plot_path = os.path.join(self.logger.output_dir, "curriculum_history.png")
            curriculum.plot_curriculum_history(save_path=curriculum_plot_path)
        
        return policy, model_coop, model_adv
    # ---------------------------
    # ----- SUPPORT METHODS -----
    # ---------------------------

    # method to linearly combine target and current policy with coefficient alfa
    def policy_combination(self, alfa, target, current):
        new_policy = policy_convex_combination(target, current, alfa)
        
        # Clear cache after policy update
        evaluator.clear_cache()
        
        return new_policy

    def model_combination(self, beta, target, current):
        new_model = model_convex_combination(self.mdp.P, target, current, beta)
        self.mdp.set_model(new_model.get_rep())
        
        # Clear cache after model update
        evaluator.clear_cache()
        
        return new_model
