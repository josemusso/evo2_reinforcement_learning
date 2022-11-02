import json
import numpy as np
import pandas as pd
import config
from utils.agents import Agent
from utils.register import get_environment
from utils.files import load_model, write_results
from stable_baselines.common import set_global_seeds
from stable_baselines import logger
import argparse
import random
import tensorflow as tf
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import ACER

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_mapping(df):
    d = {}
    for i in range(0, len(df)-3, 3):
        temp = {}
        cont = 1
        for j in range(len(df.iloc[(i)])):
            if j >= 3:
                if str(df.iloc[(i)][j]) != 'nan':
                    temp[str(df.iloc[(i)][j])] = cont
                    cont += 1
        d[str(df.iloc[(i)+1][1])] = temp
    return d


def create_dict(d1, d2):
    d = {}
    for i in d1:
        if i in d2["data"]:
            var = d2["data"][i]
            value = d1[i][var]
            d[i] = value
        else:
            d[i] = 0

    d['pm_static'] = 0

    return d


def input_pipeline(json_front):
    colnames = ['col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9',
                'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19']

    df = pd.read_csv(
        'environments/evolution/evolution/envs/var2.0.csv', names=colnames, header=None)

    d = get_mapping(df)
    init_state = create_dict(d, json_front)

    return init_state


def get_best_premes(probs, preme_json):
    prob_copy = probs.copy()
    best_premes = {}
    for i in range(1, 7):
        label = 'pred_preme'+str(i)
        best_premes[label] = {}
        max = 0
        for j in range(len(probs)):
            if prob_copy[j] > max:
                max = prob_copy[j]
                index = j
        best_premes[label]['confidence'] = round(max*100, 3)
        for k in preme_json:
            if index == preme_json[k]['id']:
                preme_name = k
        best_premes[label]['preme'] = preme_name
        prob_copy[index] = 0

    return best_premes


def output_pipeline(probs):
    with open('environments/evolution/evolution/envs/premes_lower.json', 'r') as f:
        data = json.load(f)

    best_premes = get_best_premes(probs, data)
    # print(json.dumps(best_premes,indent=2))
    with open('best_premes.json', 'w') as f:
        json.dump(best_premes, f, indent=2)


def main(args):

    dic_front = {
        "data": {
            "pm_scope": "LATAM",
            "pm_actors": "Consumidor final",
            "pm_importance": "Neutral",
            "pm_affected": "Estable",
            'pm_urgency': "Este mes",
            "pm_frequency": "Semanal",
            "pm_industry": "Minería",
        }
    }

    initial_state = input_pipeline(dic_front)

    logger.configure(config.LOGDIR)

    if args.debug:
        logger.set_level(config.DEBUG)
    else:
        logger.set_level(config.INFO)

    # make environment
    env = get_environment(args.env_name)(
        verbose=args.verbose, manual=args.manual, env_test=True, initial_state=initial_state)
    env.seed(args.seed)
    set_global_seeds(args.seed)

    acer_model = load_model(env, 'best_model.zip')

    obs = env.reset()

    while True:
        input("Press Enter to generate a prediction...")
        probs = acer_model.action_probability(obs)
        output_pipeline(probs)
        action, _states = acer_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # with open('state.txt', 'w') as f:
        #    f.write(str(obs))
        env.render()


def cli() -> None:
    """Handles argument extraction from CLI and passing to main().
    Note that a separate function is used rather than in __name__ == '__main__'
    to allow unit testing of cli().
    """
    # Setup argparse to show defaults on help
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)

    parser.add_argument("--agents", "-a", nargs='+', type=str,
                        default=['human', 'human'], help="Player Agents (human, ppo version)")

    parser.add_argument("--best", "-b", action='store_true', default=False,
                        help="Make AI agents choose the best move (rather than sampling)")

    parser.add_argument("--games", "-g", type=int, default=1,
                        help="Number of games to play)")

    # parser.add_argument("--n_players", "-n", type = int, default = 3
    #               , help="Number of players in the game (if applicable)")
    parser.add_argument("--debug", "-d",  action='store_true',
                        default=False, help="Show logs to debug level")

    parser.add_argument("--verbose", "-v",  action='store_true',
                        default=False, help="Show observation on debug logging")

    parser.add_argument("--manual", "-m",  action='store_true',
                        default=False, help="Manual update of the game state on step")

    parser.add_argument("--randomise_players", "-r",  action='store_true',
                        default=False, help="Randomise the player order")
    parser.add_argument("--recommend", "-re",  action='store_true',
                        default=False, help="Make recommendations on humans turns")

    parser.add_argument("--cont", "-c",  action='store_true', default=False,
                        help="Pause after each turn to wait for user to continue")

    parser.add_argument("--env_name", "-e",  type=str,
                        default='TicTacToe', help="Which game to play?")

    parser.add_argument("--write_results", "-w",  action='store_true',
                        default=False, help="Write results to a file?")

    parser.add_argument("--seed", "-s",  type=int,
                        default=17, help="Random seed")

    # Extract args
    args = parser.parse_args()

    # Enter main
    main(args)
    return


if __name__ == '__main__':
    cli()
