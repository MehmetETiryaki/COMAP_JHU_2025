import pandas as pd

import pandas as pd

def sort_by_name(file_path: str, output_path: str) -> None:

    df = pd.read_csv(file_path)

    # Step 1: Sort by name
    name_column = "Name"  # Replace with the column name for names
    if name_column not in df.columns:
        raise ValueError(f"'{name_column}' column is missing from the dataset.")
    
    df = df.sort_values(by=name_column)

    # Save the result to the output path
    df.to_csv(output_path, index=False)

def map_ids(file_path: str, output_path: str) -> None:
    df = pd.read_csv(file_path)

    # Step 2: Map names to unique IDs
    name_column = "Name"  # Replace with the column name for names
    year_column = "Year"  # Replace with the column name for years
    noc_column = "NOC"  # Replace with the column name for country codes
    if year_column not in df.columns or noc_column not in df.columns:
        raise ValueError(f"'{year_column}' or '{noc_column}' column is missing from the dataset.")
    
    # Sort the DataFrame by name, NOC, and year to make processing easier
    df = df.sort_values(by=[name_column, noc_column, year_column])

    # Create a new ID column
    current_id = 0
    previous_name = None
    previous_year = None
    previous_noc = None

    def assign_id(row):
        nonlocal current_id, previous_name, previous_year, previous_noc
        name, year, noc = row[name_column], row[year_column], row[noc_column]
        # Assign a new ID if the name, NOC, or year difference conditions are met
        if (name != previous_name or
            noc != previous_noc or
            (previous_year is not None and abs(year - previous_year) >= 36)):
            current_id += 1
        previous_name = name
        previous_year = year
        previous_noc = noc
        return current_id

    df["ID"] = df.apply(assign_id, axis=1)

    # Step 1: Rearrange columns
    # Move the ID column to the front, don't drop the name column
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("ID")))
    df = df[cols]

    # Save the result to the output path
    df.to_csv(output_path, index=False)


sort_by_name("summerOly_athletes.csv", "sorted.csv")
map_ids("sorted.csv", "test.csv")