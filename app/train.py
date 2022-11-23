# docker-compose exec app python3 train.py -r -e butterfly

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import argparse
import time
#from shutil import copyfile
from mpi4py import MPI

from stable_baselines3 import A2C

#from sb3_contrib import TRPO, RecurrentPPO
#from sb3_contrib import ARS,MaskablePPO,QRDQN,TQC
#from stable_baselines3 import DDPG,DQN,SAC,TD3
#from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, ProgressBarCallback, EveryNTimesteps
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

#from stable_baselines3.common.logger import Logger

#from utils.callbacks import SelfPlayCallback
from utils.files import reset_logs, reset_models
from utils.register import get_environment
from utils.selfplay import selfplay_wrapper

import config

def main(args):

  # set up logger
  new_logger = configure(config.LOGDIR, ["stdout", "csv", "tensorboard"])

  device = sb3.common.utils.get_device(device='auto')
  print(device)

  rank = MPI.COMM_WORLD.Get_rank()

  model_dir = os.path.join(config.MODELDIR, args.env_name)

  if rank == 0:
    try:
      os.makedirs(model_dir)
    except:
      pass
    reset_logs(model_dir)
    if args.reset:
      reset_models(model_dir)

  workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()

  #Logger.info('\nSetting up the selfplay training environment opponents...')
  base_env = get_environment(args.env_name)
  env = selfplay_wrapper(base_env)(opponent_type = args.opponent_type, verbose = args.verbose)
  env.seed(workerseed)

  
  #CustomPolicy = get_network_arch(args.env_name)

  params = { 
        'verbose':1,
        'tensorboard_log':config.LOGDIR,
        'seed':workerseed,
        #'use_rms_prop' :False,
        #'use_sde':True,
        #'normalize_advantage':True,
  }

  time.sleep(5) # allow time for the base model to be saved out when the environment is created

  #Logger.info('\Creating model to train...')
  model = A2C("MlpPolicy", 
              env, 
              device=device,
              #policy_kwargs=policy_kwargs, 
              **params)

  #Callbacks
  #Logger.info('\nSetting up the selfplay evaluation environment opponents...')

  # Stop training if there is no improvement after more than 5 evaluations
  stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)

  eval_freq = 5000

  #print("eval_freq:",eval_freq)

  callback_args = {
    #'eval_env': selfplay_wrapper(base_env)(opponent_type = args.opponent_type, verbose = args.verbose),
    'eval_env': Monitor(env=env, filename=config.LOGDIR, allow_early_resets=True),
    'callback_after_eval' : stop_train_callback,
    'best_model_save_path' : config.TMPMODELDIR,
    'log_path' : config.LOGDIR,
    'eval_freq' : eval_freq,
    'n_eval_episodes' : args.n_eval_episodes,
    'deterministic' : True,
    'render' : True,
    'verbose' : 0
  }

  eval_callback = EvalCallback(**callback_args)

  #Logger.info('\nSetup complete - commencing learning...\n')

  model.set_logger(new_logger)

  model.learn(total_timesteps=1000000, reset_num_timesteps = False, tb_log_name="tb", callback=eval_callback, progress_bar=True)

  model.save("logs/model")

  env.close()
  del env


def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)


  parser.add_argument("--reset", "-r", action = 'store_true', default = False
                , help="Start retraining the model from scratch")
  parser.add_argument("--opponent_type", "-o", type = str, default = 'mostly_best'
              , help="best / mostly_best / random / base / rules - the type of opponent to train against")
  parser.add_argument("--debug", "-d", action = 'store_true', default = False
              , help="Debug logging")
  parser.add_argument("--verbose", "-v", action = 'store_true', default = False
              , help="Show observation in debug output")
  parser.add_argument("--rules", "-ru", action = 'store_true', default = False
              , help="Evaluate on a ruled-based agent")
  parser.add_argument("--best", "-b", action = 'store_true', default = False
              , help="Uses best moves when evaluating agent against rules-based agent")
  parser.add_argument("--env_name", "-e", type = str, default = 'tictactoe'
              , help="Which gym environment to train in: tictactoe, connect4, sushigo, butterfly, geschenkt, frouge")
  parser.add_argument("--seed", "-s",  type = int, default = 17
            , help="Random seed")
  parser.add_argument("--eval_freq", "-ef",  type = int, default = 10240
            , help="How many timesteps should each actor contribute before the agent is evaluated?")
  parser.add_argument("--n_eval_episodes", "-ne",  type = int, default = 100
            , help="How many episodes should each actor contirbute to the evaluation of the agent")
  parser.add_argument("--threshold", "-t",  type = float, default = 0.2
            , help="What score must the agent achieve during evaluation to 'beat' the previous version?")
  parser.add_argument("--gamma", "-g",  type = float, default = 0.99
            , help="The value of gamma in PPO")
  parser.add_argument("--timesteps_per_actorbatch", "-tpa",  type = int, default = 1024
            , help="How many timesteps should each actor contribute to the batch?")
  parser.add_argument("--clip_param", "-c",  type = float, default = 0.2
            , help="The clip paramater in PPO")
  parser.add_argument("--entcoeff", "-ent",  type = float, default = 0.1
            , help="The entropy coefficient in PPO")
  parser.add_argument("--optim_epochs", "-oe",  type = int, default = 4
            , help="The number of epoch to train the PPO agent per batch")
  parser.add_argument("--optim_stepsize", "-os",  type = float, default = 0.0003
            , help="The step size for the PPO optimiser")
  parser.add_argument("--optim_batchsize", "-ob",  type = int, default = 1024
            , help="The minibatch size in the PPO optimiser")
  parser.add_argument("--lam", "-l",  type = float, default = 0.95
            , help="The value of lambda in PPO")
  parser.add_argument("--adam_epsilon", "-a",  type = float, default = 1e-05
            , help="The value of epsilon in the Adam optimiser")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()
