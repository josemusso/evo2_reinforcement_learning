# optional dependencies
from stable_baselines import logger
import config
import tensorflow as tf
import time
import os
from unidecode import unidecode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# critical dependencies
import json
import pandas as pd
from utils.register import get_environment
from utils.files import load_model
from stable_baselines.common import set_global_seeds
from stable_baselines3.common.policies import obs_as_tensor

def predict_proba(model, state):
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np

def premes_to_list(df):
    list_dict = []

    for i in range(len(df)):
        d1 = {}
        d2 = {}
        for name, value in df.iloc[i].items():
            #print(name,value)
            if 'Unnamed' not in name:
                value_clean = str(value).replace('[','')
                value_clean = value_clean.replace(']','')
                value_clean = value_clean.replace("'",'')
                value_clean = value_clean.split(',')[0]
                #print(value_clean)
                d2[name] = value_clean
        d1['data'] = d2
        list_dict.append(d1)

    return list_dict


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


def create_dict(dict_from_variables, dict_from_front):
    d = {}
    for i in dict_from_variables:
        if i in dict_from_front["data"]:
            #var = unidecode(dict_from_front["data"][i]).lower()
            var = unidecode(dict_from_front["data"][i])
            #labels = [x.lower() for x in dict_from_variables[i].keys()]
            if var in  dict_from_variables[i].keys():
                value = dict_from_variables[i][var]
            else:
                value = 0 
            d[i] = value
        else:
            d[i] = 0

    d['pm_static'] = 0

    return d


def input_pipeline(json_front):
    colnames = ['col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9',
                'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19']

    df = pd.read_csv(
        'environments/evolution/evolution/envs/tableros.csv', names=colnames, header=None)

    d = get_mapping(df)
    init_state = create_dict(d, json_front)

    return init_state


def get_best_premes(probs, preme_json):
    prob_copy = probs.copy()
    best_premes = {}
    n_rec = 15  #numero de premes recomendadas que se desea obtener
    for i in range(1, n_rec):  
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

'''
def get_best_premes(probs, preme_json):
    prob_copy = probs.copy()
    best_premes = []
    n_rec = 15  #numero de premes recomendadas que se desea obtener
    for i in range(1, n_rec):
        label = 'pred_preme'+str(i)

        max = 0
        for j in range(len(probs)):
            if prob_copy[j] > max:
                max = prob_copy[j]
                index = j
        for k in preme_json:
            if index == preme_json[k]['id']:
                preme_name = k
        best_premes.append(preme_name)
        prob_copy[index] = 0

    return best_premes
'''

def output_pipeline(probs):
    with open('environments/evolution/evolution/envs/premes.json', 'r') as f:
        data = json.load(f)

    best_premes = get_best_premes(probs, data)
    with open('best_premes.json', 'a') as f:  # aca se escribe las mejores premes con sus probabilidades en un .json pero aca habria que hacer un requests y mandarlo al servidor en vez de escribirlo localmente
        json.dump(best_premes, f, indent=2)
    return


def main(input_json):
    start_time = time.time()

    test_premes = 'C:/Users/digevo/Documents/Startups_2000.xlsx'
    premes_cov = []

    df = pd.read_excel(test_premes)
    list_test = premes_to_list(df)
    cont=0

    for state in list_test:
        print(cont)

        json_file = state
        initial_state = input_pipeline(json_file)

        seed = 17
        mc_dict = {}


        #for i in range(1):
        #logger.configure(config.LOGDIR)

        # make environment
        env = get_environment('evolution')(
            verbose=False, manual=False, env_test=True, initial_state=initial_state)
        env.seed(seed)
        set_global_seeds(seed)

        acer_model = load_model(env, 'best_model.zip')

        obs = env.reset()

        probs = acer_model.action_probability(obs)
        output_pipeline(probs)


        #print("\n--- %s seconds ---" % (time.time() - start_time))
        #input("\nPress Enter to generate next prediction...")

        #esto es solo si se desea hacer mas de una prediccion
        #action, _states = acer_model.predict(obs)
        #obs, rewards, dones, info = env.step(action)
        cont+=1




if __name__ == '__main__':
    main(0)
