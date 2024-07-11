from tqdm import tqdm
import os
import pandas as pd

df_path = "/Users/neo/Documents/DATA/mskcc_bma_abnormal.csv"

df = pd.read_csv(df_path)

# get a list of all the columns in the dataframe
columns = df.columns.tolist()

metadata = {}

for col in columns:
    metadata[col] = []

# traverse through rows of the dataframe
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
    
    fpath = row["fpath"]

    # only keep if the fpath contains "AML"
    if "AML" in fpath:
        for col in columns:
            metadata[col].append(row[col])


metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv("AML_metadata.csv", index=False)