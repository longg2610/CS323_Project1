# CS 323 Data Privacy Project 1
import pandas as pd

filename = "ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(filename)
qi_cols = ["FCVC", "CAEC", "MTRANS"]
df_qi = df[qi_cols].copy()

print(df_qi.dtypes)
print(df_qi["FCVC"].describe())
print(df_qi["CAEC"].value_counts(dropna=False))
print(df_qi["MTRANS"].value_counts(dropna=False))

df_qi["FCVC_gen"] = df_qi["FCVC"].map(lambda x: "Low" if x <= 1.3 else "High")
print(df_qi["FCVC_gen"].head())