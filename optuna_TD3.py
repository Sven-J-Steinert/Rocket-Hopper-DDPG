import optuna
import gym
import numpy as np

from TD3 import *
from hopperenv import *

import optuna

import torch
import torch.nn as nn



DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# from stable_baselines import PPO2
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines.common.cmd_util import make_vec_env

# # https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb
# from custom_env import GoLeftEnv

"""
# Hyperparameters to optimize:
Number of layers
Units per layer
Activation function

Actor learning rate
Critic learning rate
Policy frequency

Batch size
Update iteration
Buffer size
Max episodes

"""

def objective(trial):
    # Number of layers and units per layer for actor
    #actor_layers = trial.suggest_int('actor_layers', 1, 4)
    actor_neurons = trial.suggest_int('actor_neurons', 4, 512, log=True)
    # Number of layers and units per layer for critic
    #critic_layers = trial.suggest_int('critic_layers', 1, 4)
    critic_neurons = trial.suggest_int('critic_neurons', 4, 512, log=True)
    # Activation function
    #activation_fn = trial.suggest_categorical('activation_fn', ['relu', 'tanh'])
    #activation_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation_fn]
    # Learning rates
    lr_actor = trial.suggest_float('lr_actor', 1e-5, 1e-1,log=True)
    lr_critic = trial.suggest_float('lr_critic', 1e-5, 1e-1,log=True)
    
    # Target network
    tau = trial.suggest_float('tau', 1e-3, 1e-1,log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.99,log=True)

    # Batch size
    batch_size = trial.suggest_int('batch_size', 32, 256, log=True)
    
    '''
    hyperparameters = {
        'actor_layers': actor_layers,
        'actor_neurons': actor_neurons,
        'critic_layers': critic_layers,
        'critic_neurons': critic_neurons,
        'activation_fn': activation_fn,
        'lr_actor': lr_actor,
        'lr_critic': lr_critic,
        'tau': tau,
        'gamma': gamma,
        'batch_size': batch_size
    }
    '''
    
    hyperparameters = {
        'actor_neurons': actor_neurons,
        'critic_neurons': critic_neurons,
        'lr_actor': lr_actor,
        'lr_critic': lr_critic,
        'tau': tau,
        'gamma': gamma,
        'batch_size': batch_size
    }

    accuary = run(hyperparameters)

    return accuary


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize',storage="sqlite:///db.sqlite3")  # or 'maximize' depending on your metric
    study.optimize(objective, n_trials=1)  # You can adjust the number of trials

    # Print the best hyperparameters
    print('Best hyperparameters:', study.best_params)

    # Retrieve the best hyperparameters
    best_hyperparams = study.best_params

    # Create an instance of your TD3 model with the best hyperparameters
    run(best_hyperparams, final=True)
