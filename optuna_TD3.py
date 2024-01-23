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
    actor_layers = trial.suggest_int('actor_layers', 1, 4)
    actor_neurons = trial.suggest_int('actor_neurons', 4, 512, log=True)
    # Number of layers and units per layer for critic
    critic_layers = trial.suggest_int('critic_layers', 1, 4)
    critic_neurons = trial.suggest_int('critic_neurons', 4, 512, log=True)
    # Activation function
    activation_fn = trial.suggest_categorical('activation_fn', ['relu', 'tanh'])
    activation_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation_fn]
    # Learning rates
    lr_actor = trial.suggest_loguniform('lr_actor', 1e-5, 1e-1,log=True)
    lr_critic = trial.suggest_loguniform('lr_critic', 1e-5, 1e-1,log=True)
    
    # Target network
    tau = trial.suggest_float('tau', 1e-3, 1e-1,log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.99,log=True)

    # Batch size
    batch_size = trial.suggest_int('batch_size', 32, 256, log=True)

    return {
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

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
    


# XXX TBC
    
def objective(trial):


def create_model(trial, n_layers, in_size):

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, 10))

    return nn.Sequential(*layers)


def optimize_ppo2(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo2(trial)
    env = make_vec_env(lambda: GoLeftEnv(), n_envs=16, seed=0)
    model = PPO2('MlpPolicy', env, verbose=0, nminibatches=1, **model_params)
    model.learn(10000)
    mean_reward, _ = evaluate_policy(model, GoLeftEnv(), n_eval_episodes=10)

    return -1 * mean_reward





# Create Optuna study and perform optimization
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(),
    direction='maximize'
)

try:
    study.optimize(optimize_agent, n_trials=100, n_jobs=4)
except KeyboardInterrupt:
    print('Interrupted by keyboard.')


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')  # or 'maximize' depending on your metric
    study.optimize(objective, n_trials=100)  # You can adjust the number of trials

    # Print the best hyperparameters
    print('Best hyperparameters:', study.best_params)

    # Retrieve the best hyperparameters
    best_hyperparams = study.best_params

    # Create an instance of your TD3 model with the best hyperparameters
    best_td3_model = TD3(state_dim, action_dim, H1, H2, H2, 'relu', best_hyperparams['learning_rate_actor'],
                         best_hyperparams['learning_rate_critic'], best_hyperparams['policy_freq'],
                         best_hyperparams['batch_size'], best_hyperparams['update_iteration'],
                         best_hyperparams['buffer_size'], best_hyperparams['max_episodes'])

    # Train the best model
    train_td3_model(best_td3_model, custom_env, final_training=True)    