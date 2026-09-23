import pyarrow.ipc as ipc
import pandas as pd

input_file = "data-00000-of-00001.arrow"
output_file = "edurabsa_acd_train.csv"

print("Reading Arrow dataset...")

with open(input_file, "rb") as f:
    reader = ipc.open_stream(f)
    table = reader.read_all()

df = table.to_pandas()

print("\nDataset loaded!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nTask types:")
print(df["task_type"].value_counts())

print("\nSaving CSV...")
df.to_csv(output_file, index=False)

print(f"\nDone!")
print(f"Saved as: {output_file}")