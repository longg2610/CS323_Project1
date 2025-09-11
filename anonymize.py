import pandas as pd
import numpy as np


df = pd.read_csv("./ObesityDataSet_raw_and_data_sinthetic.csv")

#Quasi-identifiers: Gender, Age, Height, Weight, CH2O (0-1, 2-3), SCC(), FAF, TUE, MTRANS

#Anonymize: Age, Height, Weight (QID set)
#Suppress FAVC, FCVC, NCP, CAEC (trash data)
#Sensitive Data: family health (family_history_with_overweight), smoking (SMOKE), alcohol (CALC), obesity (NObeyesdad)

#penalty doesnt follow range linearly

df = df.drop("FAVC", axis = 1)
df = df.drop("FCVC", axis = 1)
df = df.drop("NCP", axis = 1)
df = df.drop("CAEC", axis = 1)


age_buckets_0 = list(range(70, 9, -5))
age_buckets_1 = list(range(70, 9, -30))
age_buckets_2 = list(range(70, 9, -50))
age_buckets_3 = list(range(70, 9, -60))


height_buckets_0 = np.arange(1.9, 1.39, -0.01)
height_buckets_1 = np.arange(1.9, 1.39, -0.05)
height_buckets_2 = np.arange(1.9, 1.39, -0.10)
height_buckets_3 = np.arange(1.9, 1.39, -0.3)

# height_buckets_0 = [1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45, 1.4]
# height_buckets_1 = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4]
# height_buckets_2 = [1.9, 1.7, 1.5]
# height_buckets_3 = [1.9, 1.5]


weight_buckets_0 = list(range(170, 29, -10))
weight_buckets_1 = list(range(170, 29, -70))
weight_buckets_2 = list(range(170, 29, -120))
weight_buckets_3 = list(range(170, 29, -150))

def get_age_buckets(buckets_no):
    if buckets_no == 0:
        age_buckets = age_buckets_0
    elif buckets_no == 1:
        age_buckets = age_buckets_1
    elif buckets_no == 2:
        age_buckets = age_buckets_2
    else:
        age_buckets = age_buckets_3
    return age_buckets

def get_height_buckets(buckets_no):
    if buckets_no == 0:
        height_buckets = height_buckets_0
    elif buckets_no == 1:
        height_buckets = height_buckets_1
    elif buckets_no == 2:
        height_buckets = height_buckets_2
    else:
        height_buckets = height_buckets_3
    return height_buckets

def get_weight_buckets(buckets_no):
    if buckets_no == 0:
        weight_buckets = weight_buckets_0
    elif buckets_no == 1:
        weight_buckets = weight_buckets_1
    elif buckets_no == 2:
        weight_buckets = weight_buckets_2
    else:
        weight_buckets = weight_buckets_3
    return weight_buckets


def age_generalize(age, age_buckets):
    for bucket in age_buckets:
        if age >= bucket:
            return bucket

def height_generalize(height, height_buckets):
    for bucket in height_buckets:
        if height >= bucket:
            return bucket

def weight_generalize(weight, weight_buckets):
    for bucket in weight_buckets:
        if weight >= bucket:
            return bucket


def check_k_anonymity(df, k, ages, heights, weights):
    penalty = 0
    for age in ages:
        for height in heights:
            for weight in weights:
                count = ((df["Age"] == age) & (df["Height"] == height) & (df["Weight"] == weight)).sum()
                # print(age)
                penalty += ((count-1) * count)
                if count < k and count > 0:
                    return (False, -1)
                
    return (True, penalty)


penalty_dict = {}

def anonymize(df, k, pass_vector):
    age_buckets = get_age_buckets(pass_vector[0])
    height_buckets = get_height_buckets(pass_vector[1])
    weight_buckets = get_weight_buckets(pass_vector[2])

    # generalize based on column
    df["Age"] = df["Age"].apply(age_generalize, args=(age_buckets,))
    df["Height"] = df["Height"].apply(height_generalize, args=(height_buckets,))
    df["Weight"] = df["Weight"].apply(weight_generalize, args=(weight_buckets,))

    # print(df.head(20))

    check_result = check_k_anonymity(df, k, age_buckets, height_buckets, weight_buckets)       #if check fails, return
    if check_result[0] == False:
        print(f"Not k-anonymized, the unsatisfying QID is: {pass_vector[0]} {pass_vector[1]} {pass_vector[2]}")

    else:
        print(f"{pass_vector[0]} {pass_vector[1]} {pass_vector[2]} k-anonymized with a penalty of {check_result[1]}")


    penalty_dict[(pass_vector[0], pass_vector[1], pass_vector[2])] = check_result[1]         # negative penalty means failed

    #return on max depth
    if sum(pass_vector) == 2:
        return
    
    
    for i in range(len(pass_vector)):
        copy = pass_vector.copy()
        copy[i] = copy[i] + 1
        anonymize(df, k, copy)

anonymize(df, 2, [0,0,0])

positive_items = {k: v for k, v in penalty_dict.items() if v > 0}
min_key = min(positive_items, key=positive_items.get)
print(min_key, penalty_dict[min_key])


# # print(df.head())
# print(df.columns)

# #Age ranges: 14-20, 21-29, 30-39, 40-49, 50-59, 60-61 (6)
# print("Age min:", df["Age"].min())
# print("Age max:", df["Age"].max())

# #Height ranges: 1.40-1.49, 1.50-1.59, 1.60-1.69, 1.70-1.79, 1.80-1.89, 1.90-1.99 (6)
# print("Height min:", df["Height"].min())
# print("Height max:", df["Height"].max())

# #Weight ranges: 30-49, 50-69, 70-89, 90-109, 110-129, 130-149, 150-169, 170-189 (8)
# print("Weight min:", df["Weight"].min())
# print("Weight max:", df["Weight"].max())


