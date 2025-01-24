import pandas as pd

# Step 1: Read the CSV file
input_file = "summerOly_athletes.csv"  # Replace with your input file path
output_file = "summerOly_athletes_ID.csv"  # Replace with your output file path
df = pd.read_csv(input_file)

# Step 2: Map names to unique IDs
name_column = "Name"  # Replace with the column name for names
df["ID"] = df[name_column].astype('category').cat.codes
# move the ID column to the front, then drop the name column
df = df[["ID"] + [col for col in df.columns if col != "ID"]]

# Step 3: Save the new CSV file
df.to_csv(output_file, index=False)

print(f"Encoded names to IDs and saved to {output_file}.")