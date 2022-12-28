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
    random_number = int(np.random.uniform(0, 7))
    chosen_assesment_index = int(random_number)
    chosen_assessment_number = chosen_assesment_index + 1

    # for preme_name in premes_list:
    print(
        f"Assessment number {chosen_assessment_number} was chosen, index {chosen_assesment_index}"
    )

    print(
        f"Generating effects of previous assessments: {np.arange(chosen_assesment_index)}"
    )
    preme_name = premes_list[chosen_assesment_index]
    premes_dict = {
        "pr_initial_assessment": {
            "id": int(df[df["nombre_tecnico_preme"] == preme_name]["id"].values[0]),
            "name": str(
                df[df["nombre_tecnico_preme"] == preme_name]["nombre_preme"].values[0]
            ),
            "repetitive": int(
                df[df["nombre_tecnico_preme"] == preme_name]["repetitividad"].values[0]
            ),
        }
    }
    # randomly choose effects
    # each has 0.5 chance of being chosen
    all_effects = []
    for i in np.arange(chosen_assesment_index + 1):
        chosen_effects = []
        previous_preme_name = premes_list[i]
        for effect in effects_serie[previous_preme_name]:
            # only apply random to last assessment, to simulate all other assesments 100% complete
            if i == chosen_assesment_index:
                rand = int(np.random.uniform(0, 2))
            else:
                rand = 1
            if rand:
                chosen_effects.append(effect)
        print(
            f"{previous_preme_name}: Chosen {len(chosen_effects)} effects out of {len(effects_serie[previous_preme_name])}"
        )
        all_effects.extend(chosen_effects)

    print(f"Total effects: {len(all_effects)}")

    premes_dict["pr_initial_assessment"]["effects"] = all_effects

    premes_dict["pr_initial_assessment"]["restrictions"] = []
    premes_dict["pr_initial_assessment"]["id_preme"] = int(
        df[df["nombre_tecnico_preme"] == preme_name]["id_preme"].values[0]
    )

    # mapear json
    # print(premes_dict)
    # print(json.dumps(premes_dict, indent=4, ensure_ascii=False))

    with open("random_initial_assesment.json", "w") as fp:
        json.dump(premes_dict, fp, indent=4, ensure_ascii=False)
    print(f"Initial assesment generated and written at random_initial_assesment.json")

    return


premes_csv_to_json("assessments.csv")
