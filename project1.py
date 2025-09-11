# CS 323 Data Privacy Project 1
import pandas as pd

filename = "ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(filename)
qi_cols = ["FCVC", "CAEC", "MTRANS"]
df_qi = df[qi_cols].copy()

# print(df_qi.dtypes)
# print(df_qi["FCVC"].describe())
# print(df_qi["CAEC"].value_counts(dropna=False))
# print(df_qi["MTRANS"].value_counts(dropna=False))

df_qi["FCVC_gen"] = df_qi["FCVC"].map(lambda x: "Low" if x <= 1.3 else "High")
print(df_qi["FCVC_gen"].head())

def generalize_CAEC(x):
    if x in["Always", 'Frequently']:
        return "Often"
    elif x in ["Sometimes", "No"]:
        return "Rarely"
    else:
        return None

df_qi["CAEC_gen"] = df_qi["CAEC"].map(generalize_CAEC)
print(df_qi["CAEC_gen"].head())

def generalize_MTRANS(x):
    if x in ["Automobile", "Motorbike"]:
        return "Private"
    elif x in ["Public_Transportation", "Walking", "Bike"]:
        return "Non-private"
    else:
        return None
    
df_qi["MTRANS_gen"] = df_qi["MTRANS"].map(generalize_MTRANS)
print(df_qi["MTRANS_gen"].head())