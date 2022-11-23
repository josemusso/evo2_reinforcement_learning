# optional dependencies
from stable_baselines import logger
import config
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# critical dependencies
import json
import pandas as pd
from utils.register import get_environment
from utils.files import load_model
from stable_baselines.common import set_global_seeds


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
    with open('best_premes.json', 'w') as f:
        json.dump(best_premes, f, indent=2)


def main(input_json):
    start_time = time.time()

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

    json_file = dic_front
    #input_json = json_file

    initial_state = input_pipeline(json_file)

    seed = 17

    #logger.configure(config.LOGDIR)

    # make environment
    env = get_environment('evolution')(
        verbose=False, manual=False, env_test=True, initial_state=initial_state)
    env.seed(seed)
    set_global_seeds(seed)

    acer_model = load_model(env, 'best_model.zip')

    obs = env.reset()

    while True:
        probs = acer_model.action_probability(obs)
        output_pipeline(probs)
        print("\n--- %s seconds ---" % (time.time() - start_time))
        input("\nPress Enter to generate next prediction...")
        action, _states = acer_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # with open('state.txt', 'w') as f:
        #    f.write(str(obs))
        env.render()


if __name__ == '__main__':
    main(0)
