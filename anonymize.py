import pandas as pd
import numpy as np
import math

df = pd.read_csv("./ObesityDataSet_raw_and_data_sinthetic.csv")

# filters out edge values
df = df[
    (df["Age"].between(16, 50)) &
    (df["Height"].between(1.5, 1.9)) &
    (df["Weight"].between(60, 140))
]

print(len(df))

print("Age min:", df["Age"].min())
print("Age max:", df["Age"].max())

print("Height min:", df["Height"].min())
print("Height max:", df["Height"].max())

print("Weight min:", df["Weight"].min())
print("Weight max:", df["Weight"].max())


def make_buckets(col_min, col_max, step, use_numpy=False):
    # Round min and max to align with step
    if use_numpy:
        # For floats → floor min, ceil max
        start = math.ceil(col_max / step) * step
        end = math.floor(col_min / step) * step
        buckets = np.arange(start, end - step, -step)
        if buckets[-1] > col_min:   # ensure last bucket <= min
            buckets = np.append(buckets, col_min)
        return buckets
    else:
        # For integers
        start = math.ceil(col_max / step) * step
        end = math.floor(col_min / step) * step
        buckets = list(range(start, end - 1, -step))
        if buckets[-1] > col_min:   # ensure last bucket == min
            buckets.append(int(col_min))
        return buckets


# generalization buckets hardcoded
age_min, age_max = df["Age"].min(), df["Age"].max()
age_buckets_0 = make_buckets(age_min, age_max, 5)
age_buckets_1 = make_buckets(age_min, age_max, 10)
age_buckets_2 = make_buckets(age_min, age_max, 20)
age_buckets_3 = make_buckets(age_min, age_max, 50)

print(age_buckets_0)
print(age_buckets_1)
print(age_buckets_2)
print(age_buckets_3)


height_min, height_max = df["Height"].min(), df["Height"].max()
height_buckets_0 = np.round(make_buckets(height_min, height_max, 0.1, use_numpy=True), 2)
height_buckets_1 = np.round(make_buckets(height_min, height_max, 0.3, use_numpy=True), 2)
height_buckets_2 = np.round(make_buckets(height_min, height_max, 0.5, use_numpy=True), 2)
height_buckets_3 = np.round(make_buckets(height_min, height_max, 0.7, use_numpy=True), 2)

print(height_buckets_0)
print(height_buckets_1)
print(height_buckets_2)
print(height_buckets_3)

weight_min, weight_max = df["Weight"].min(), df["Weight"].max()
weight_buckets_0 = make_buckets(weight_min, weight_max, 5)
weight_buckets_1 = make_buckets(weight_min, weight_max, 20)
weight_buckets_2 = make_buckets(weight_min, weight_max, 50)
weight_buckets_3 = make_buckets(weight_min, weight_max, 100)

print(weight_buckets_0)
print(weight_buckets_1)
print(weight_buckets_2)
print(weight_buckets_3)


def age_generalize(age, age_buckets):
    for bucket in age_buckets:
        if age >= bucket:
            return bucket
    # print("last bucket: ", age_buckets[-1], " age: ", age)
    return age_buckets[-1]

def height_generalize(height, height_buckets):
    for bucket in height_buckets:
        if height >= bucket:
            return bucket
    # print("last bucket: ", height_buckets[-1], " height: ", height)
    return height_buckets[-1]

def weight_generalize(weight, weight_buckets):
    for bucket in weight_buckets:
        if weight >= bucket:
            return bucket
    return weight_buckets[-1]



def check_k_anonymity(df, k, ages, heights, weights):
    penalty = 0
    for age in ages:
        for height in heights:
            for weight in weights:
                count = ((df["Age"] == age) & (df["Height"] == height) & (df["Weight"] == weight)).sum()
                penalty += count ** 2
                if count < k and count > 0:
                    # print("QID (", age, height, weight, ") appeared", count, "times")
                    return (False, -1)
    return (True, penalty)



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

penalty_dict = {}
original = df.copy(deep=True)
def anonymize(df, k, pass_vector):
    if (pass_vector[0], pass_vector[1], pass_vector[2]) in penalty_dict:    #already tried
        return

    age_buckets = get_age_buckets(pass_vector[0])
    height_buckets = get_height_buckets(pass_vector[1])
    weight_buckets = get_weight_buckets(pass_vector[2])

    # generalize based on column
    df["Age"] = df["Age"].apply(age_generalize, args=(age_buckets,))
    df["Height"] = df["Height"].apply(height_generalize, args=(height_buckets,))
    df["Weight"] = df["Weight"].apply(weight_generalize, args=(weight_buckets,))

    # print(df.head(10))

    check_result = check_k_anonymity(df, k, age_buckets, height_buckets, weight_buckets)    
    if check_result[0] == False:
        print(f"{pass_vector[0]} {pass_vector[1]} {pass_vector[2]} did NOT k-anonymize")

    else:
        print(f"{pass_vector[0]} {pass_vector[1]} {pass_vector[2]} k-anonymized with a penalty of {check_result[1]}")


    penalty_dict[(pass_vector[0], pass_vector[1], pass_vector[2])] = check_result[1]         # negative penalty means failed

    #return on max depth
    # if sum(pass_vector) == 3:
    #     return
    
    for i in range(len(pass_vector)):
        copy = pass_vector.copy()
        if (copy[i] + 1) <= 3:      # max depth
            copy[i] = copy[i] + 1
            df = original.copy(deep=True)
            anonymize(df, k, copy)


def main():
    k = int(input("Enter the value of k for k-anonymity: "))

    # perform search for the best generalization node
    anonymize(df, k, [0,0,0])      # 2, 15, 45

    # get the node with lowest penalty
    positive_items = {k: v for k, v in penalty_dict.items() if v > 0}
    min_key = min(positive_items, key=positive_items.get)
    print(min_key, penalty_dict[min_key])

    age_buckets = get_age_buckets(min_key[0])
    height_buckets = get_height_buckets(min_key[1])
    weight_buckets = get_weight_buckets(min_key[2])
    
    original["Age"] = original["Age"].apply(age_generalize, args=(age_buckets,))
    original["Height"] = original["Height"].apply(height_generalize, args=(height_buckets,))
    original["Weight"] = original["Weight"].apply(weight_generalize, args=(weight_buckets,))

    # final check
    group_counts = original.groupby(['Age', 'Height', 'Weight']).size()    
    violating_groups = group_counts[group_counts < k]
    if len(violating_groups) > 0:
        print(f"Dataframe does NOT satisfy {k}-anonymity.")
        print("Violating QID combinations and their counts:")
        print(violating_groups)
    else:
        print(f"Dataframe satisfies {k}-anonymity.")

    original.to_csv("output.csv", index=False)

main()

