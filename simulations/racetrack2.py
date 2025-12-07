import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import os

from algorithm.model_chooser import *
from algorithm.spmi import SPMI
from utils.metrics_evaluator import EvaluationOptions, IterationMetricsEvaluator

from algorithm.policy_chooser import *
from envs.racetrack_simulator import RaceTrackConfigurableEnv
from utils.uniform_policy import UniformPolicy
from utils.tabular import *

track = 'T1'
simulation_name = 'racetrack2_' + track
dir_path = "./data/" + simulation_name

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

mdp = RaceTrackConfigurableEnv(track_file=track, initial_configuration=[0.5, 0.5, 0, 0], pfail=0.07)

original_model = copy.deepcopy(mdp.P)
ref_model = TabularModel(original_model, mdp.nS, mdp.nA)

uniform_policy = UniformPolicy(mdp)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

model_set = [TabularModel(mdp.P_highspeed_noboost, mdp.nS, mdp.nA),
             TabularModel(mdp.P_lowspeed_noboost, mdp.nS, mdp.nA)]

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = SetModelChooser(model_set, mdp.nS, mdp.nA)

eps = 0.0
max_iter = 1000 # default: 30000
metrics_opts = EvaluationOptions(
    state_coverage=True,
    state_entropy=True,
    reward_diversity=True,
    model_divergence=True
)
metrics_evaluator = IterationMetricsEvaluator(
    mdp,
    options=metrics_opts,
    reference_model=ref_model
)

spmi = SPMI(
    mdp,
    eps,
    policy_chooser,
    model_chooser,
    max_iter=max_iter,
    persistent=True,
    delta_q=1,
    metrics_evaluator=metrics_evaluator,
)

#-------------------------------------------------------------------------------
#SPMI
spmi.spmi(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spmi.csv')
metrics_evaluator.save(dir_path, "metrics_spmi.csv")


#-------------------------------------------------------------------------------
#SPMI-sup
# mdp.set_initial_configuration(original_model)
# spmi.spmi_sup(initial_policy, initial_model)

# spmi.logger.save(dir_path, 'spmi_sup.csv')
# metrics_evaluator.save(dir_path, "metrics_spmi_sup.csv")

# #-------------------------------------------------------------------------------
# #SPMI-alt
# mdp.set_initial_configuration(original_model)
# spmi.spmi_alt(initial_policy, initial_model)

# spmi.logger.save(dir_path, 'spmi_alt.csv')
# metrics_evaluator.save(dir_path, "metrics_spmi_alt.csv")

# #-------------------------------------------------------------------------------
# #SPI+SMI
# mdp.set_initial_configuration(original_model)
# spmi.spi_smi(initial_policy, initial_model)

# spmi.logger.save(dir_path, 'spi_smi.csv')
# metrics_evaluator.save(dir_path, "metrics_spi_smi.csv")

# #-------------------------------------------------------------------------------
# #SMI+SPI
# mdp.set_initial_configuration(original_model)
# spmi.smi_spi(initial_policy, initial_model)

# spmi.logger.save(dir_path, 'smi_spi.csv')
# metrics_evaluator.save(dir_path, "metrics_smi_spi.csv")
