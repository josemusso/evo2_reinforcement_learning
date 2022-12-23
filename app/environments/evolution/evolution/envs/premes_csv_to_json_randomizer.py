from operator import index
import numpy as np
import pandas as pd
import json
import ast


def delete_nans(preme_dict):
    c = 0
    for key, value in preme_dict.items():
        for var in value:
            # print(var)
            if var["var_name"] == "NaN":
                c += 1
                # print(var)
                value.remove(var)
    return c


def premes_csv_to_json(filepath):

    df = pd.read_csv(filepath, header=1, encoding="utf-8")

    # dejar solo las que tienen el flag en_uso
    # df = df[df['en_uso']==1]

    # fill nans produced by merging cells with previous values
    df["nombre_tecnico_preme"] = df["nombre_tecnico_preme"].fillna(method="ffill")
    df["nombre_preme"] = df["nombre_preme"].fillna(method="ffill")
    # df['en uso'] = df['en uso'].fillna(method='ffill')
    df["id"] = df["id"].fillna(method="ffill")
    df["id_preme"] = df["id_preme"].fillna(method="ffill")
    df["repetitividad"] = df["repetitividad"].fillna(method="ffill")
    # df.set_index('nombre_tecnico_preme')

    # cant handle real NaN, replaced by "NaN"
    df = df.fillna(0)
    # print(df.head(20))

    # generate list of premes with multiple restrictions or effects
    restrictions_serie = df.groupby("nombre_tecnico_preme")[
        ["var_name", "condition", "value"]
    ].apply(lambda x: x.to_dict(orient="records"))
    df = df.rename(
        columns={
            "var_name": "var_name.2",
            "value": "value.2",
            "var_name.1": "var_name",
            "value.1": "value",
        }
    )
    effects_serie = df.groupby("nombre_tecnico_preme")[
        ["var_name", "operator", "value", "dimension", "inc_dim"]
    ].apply(lambda x: x.to_dict(orient="records"))
    # print(restrictions_serie)
    # print(effects_serie)
    cont_restrictions = delete_nans(restrictions_serie)
    while cont_restrictions > 0:
        # print('rest')
        cont_restrictions = delete_nans(restrictions_serie)

    cont_effects = delete_nans(effects_serie)
    while cont_effects > 0:
        # print('eff')
        cont_effects = delete_nans(effects_serie)

    # reformatear restirctions y effects
    premes_list = list(df["nombre_tecnico_preme"].unique())
    # print(list(df['nombre_tecnico_preme'].unique()))
    # premes_dict = {key: {} for key in premes_list}

    # select which assesments gets chosen, each with equal probability
    random_number = np.random.rand(1)[0] * 7
    chosen_assesment = int(random_number)

    # for preme in premes_list:
    preme = premes_list[chosen_assesment]
    print(f"Preme {preme} was chosen")
    premes_dict = {preme: {}}

    premes_dict[preme]["id"] = int(
        df[df["nombre_tecnico_preme"] == preme]["id"].values[0]
    )
    premes_dict[preme]["name"] = str(
        df[df["nombre_tecnico_preme"] == preme]["nombre_preme"].values[0]
    )
    repetitity = int(df[df["nombre_tecnico_preme"] == preme]["repetitividad"].values[0])
    premes_dict[preme]["repetitive"] = repetitity
    # if repetitity == 1:
    #    premes_dict[preme]["repetitive"] = True
    # else:
    #    premes_dict[preme]["repetitive"] = False

    # randomly choose effects
    # each has 0.5 chance of being chosen
    chosen_effects = []
    for effect in effects_serie[preme]:
        if int(np.random.rand(1)[0] * 2):
            chosen_effects.append(effect)
    print(f"Chosen {len(chosen_effects)} effects out of {len(effects_serie[preme])}")
    premes_dict[preme]["effects"] = chosen_effects

    premes_dict[preme]["restrictions"] = restrictions_serie[preme]
    premes_dict[preme]["id_preme"] = int(
        df[df["nombre_tecnico_preme"] == preme]["id_preme"].values[0]
    )

    # mapear json
    # print(premes_dict)
    # print(json.dumps(premes_dict, indent=4, ensure_ascii=False))

    with open("random_initial_assesment.json", "w") as fp:
        json.dump(premes_dict, fp, indent=4, ensure_ascii=False)
    print(f"Initial assesment generated and written at random_initial_assesment.json")

    return


premes_csv_to_json("assessments.csv")
