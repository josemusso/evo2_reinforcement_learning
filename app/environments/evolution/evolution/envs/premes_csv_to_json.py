from operator import index
import numpy as np
import pandas as pd
import json
import ast


def premes_csv_to_json(filepath):

    df = pd.read_csv(filepath,header=1, encoding='utf-8')
    df = df[df['en_uso']==1]

    # fill nans produced by merging cells with previous values
    df['nombre_tecnico_preme'] = df['nombre_tecnico_preme'].fillna(method='ffill')
    df['nombre_preme'] = df['nombre_preme'].fillna(method='ffill')
    # df['en uso'] = df['en uso'].fillna(method='ffill')
    df['id'] = df['id'].fillna(method='ffill')
    # df.set_index('nombre_tecnico_preme')
    
    # cant handle real NaN, replaced by "NaN"
    df = df.fillna("NaN")
    print(df.head(20))

    # generate list of premes with multiple restrictions or effects
    restrictions_serie = df.groupby('nombre_tecnico_preme')[['var_name','condition','value']].apply(lambda x: x.to_dict(orient='records'))
    df = df.rename(columns={"var_name": "var_name.2", "value": "value.2","var_name.1": "var_name", "value.1": "value"})
    effects_serie = df.groupby('nombre_tecnico_preme')[['var_name','operator','value']].apply(lambda x: x.to_dict(orient='records'))
    # print(restrictions_serie)
    # print(effects_serie)    

    # reformatear restirctions y effects
    premes_list = list(df['nombre_tecnico_preme'].unique())
    print(list(df['nombre_tecnico_preme'].unique()))
    premes_dict = {key: {} for key in premes_list}
    for preme in premes_list:
        premes_dict[preme]["id"] = int(df[df['nombre_tecnico_preme']==preme]["id"].values[0])
        premes_dict[preme]["name"] = str(df[df['nombre_tecnico_preme']==preme]["nombre_preme"].values[0])
        premes_dict[preme]["repetitive"] = int(df[df['nombre_tecnico_preme']==preme]["repetitividad"].values[0])
        premes_dict[preme]["restrictions"] = restrictions_serie[preme]
        premes_dict[preme]["effects"] = effects_serie[preme]

    # mapear json
    print(premes_dict)
    print(json.dumps(premes_dict, indent=4, ensure_ascii=False))

        
    with open('premes_from_csv.json', 'w') as fp:
        json.dump(premes_dict,fp,indent=4,ensure_ascii=False)

    return

premes_csv_to_json("evo2_reinforcement_learning_matrices - descripcion_premes.csv")