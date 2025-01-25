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

def get_medals(input_file, output_file):

    medals_dict = {}
    already_counted = set()

    allowed_medals = ["Gold", "Silver", "Bronze"]

    df = pd.read_csv(input_file)

    df = df[df["Medal"] != "No medal"]

    for row in df.iterrows():
        sport = row[1]["Sport"]
        year = int(row[1]["Year"])
        country = row[1]["NOC"]
        medal = row[1]["Medal"]
        event = row[1]["Event"]
        team = bool(row[1]["Team_Sport"])

        if team:
            if (country, year, sport, event) in already_counted:
                continue
            else:
                already_counted.add((country, year, sport, event))

        if medal not in allowed_medals:
            raise ValueError(f"Invalid medal value '{medal}' found in the dataset.")

        if (country, year, sport) not in medals_dict:
            medals_dict[(country, year, sport)] = {"Gold": 0, "Silver": 0, "Bronze": 0}
        
        medals_dict[(country, year, sport)][medal] += 1

    medals_df = pd.DataFrame(columns=["NOC", "Year", "Sport", "Gold", "Silver", "Bronze"])

    for key, value in medals_dict.items():
        country, year, sport = key
        gold, silver, bronze = value["Gold"], value["Silver"], value["Bronze"]
        medals_df = pd.concat([medals_df, pd.DataFrame([[country, year, sport, gold, silver, bronze]], columns=medals_df.columns)], ignore_index=True)

    medals_df.to_csv(output_file, index=False)

def get_irregularities(input_file):

    df = pd.read_csv(input_file)
    df = df[df["Medal"] != "No medal"]

    year_sport_event_dict = {}
    

    for row in df.iterrows():
        year = row[1]["Year"]
        sport = row[1]["Sport"]
        event = row[1]["Event"]
        team = bool(row[1]["Team_Sport"])
        medal = row[1]["Medal"]

        if team:
            continue

        if (year, sport, event) not in year_sport_event_dict:
            year_sport_event_dict[(year, sport, event)] = {"Gold": 0, "Silver": 0, "Bronze": 0}

        if sport == "Boxing" and year == 1960 and event == "Boxing Men's Flyweight":
            print(row)
        year_sport_event_dict[(year, sport, event)][medal] += 1

    with open("irregularities.txt", "w") as f:
        for key, value in year_sport_event_dict.items():
            year, sport, event = key
            gold, silver, bronze = value["Gold"], value["Silver"], value["Bronze"]
            if gold != 1 or silver != 1 or bronze != 1:
                f.write(f"{year}, {sport}, {event}: {gold} Gold, {silver} Silver, {bronze} Bronze\n")

get_irregularities("athletes_teams.csv")