import pandas as pd
import torch

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

def get_team_sports(input_file, output_file):
    team_keyword_list = set(["Relay", "Double", "Team", "Mixed", "Two Person", "Three Person", "Four Person", "Five Person", "Six Person", "Pairs", "Fours", "Quadruple", "Sixes", "Eights",
                         "Basketball", "Baseball", "Volleyball", "Tandem", "Football", "Handball", "Hockey", "Curling", "Water Polo", "Rugby", "Rowing", "Soccer", "Softball", "Rugby", "Polo", "Cricket", "Lacrosse", "Tug-Of-War", "Group"])

    df = pd.read_csv(input_file)
    # create a column named Team_Sport that is 1 if either column "Sport" or "Event" contains a team keyword in team_keyword_list, 0 otherwise
    df["Team_Sport"] = df.apply(lambda row: 1 if any(keyword in row["Sport"] for keyword in team_keyword_list) or any(keyword in row["Event"] for keyword in team_keyword_list) else 0, axis=1)
    df.to_csv(output_file, index=False)

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
        year_sport_event_dict[(year, sport, event)][medal] += 1

    with open("irregularities.txt", "w") as f:
        for key, value in year_sport_event_dict.items():
            year, sport, event = key
            gold, silver, bronze = value["Gold"], value["Silver"], value["Bronze"]
            if gold != 1 or silver != 1 or bronze != 1:
                f.write(f"{year}| {sport}| {event}: {gold} Gold, {silver} Silver, {bronze} Bronze\n")

    with open("high_irregularities.txt", "w") as f:
        for key, value in year_sport_event_dict.items():
            year, sport, event = key
            gold, silver, bronze = value["Gold"], value["Silver"], value["Bronze"]
            if (gold != 1 or silver != 1 or bronze != 1) and not (gold == 1 and silver == 1 and bronze == 2):
                f.write(f"{year}| {sport}| {event}: {gold} Gold, {silver} Silver, {bronze} Bronze\n")

    with open("bronze_irregularities.txt", "w") as f:
        for key, value in year_sport_event_dict.items():
            year, sport, event = key
            gold, silver, bronze = value["Gold"], value["Silver"], value["Bronze"]
            if gold == 1 and silver == 1 and bronze == 2:
                f.write(f"{year}| {sport}| {event}: {gold} Gold, {silver} Silver, {bronze} Bronze\n")

def delete_irregulars(irregular_txt, input_file, output_file):

    with open(irregular_txt, "r") as f:
        irregulars = f.readlines()

    df = pd.read_csv(input_file)

    for line in irregulars:
        year, sport, event = line.split(":")[0].split("| ")
        year = int(year)
        df = df[~((df["Year"] == year) & (df["Sport"] == sport) & (df["Event"] == event))]

    df.to_csv(output_file, index=False)

def get_medal_tensor(input_file):
    num_unique_times = len(pd.read_csv(input_file)["Year"].unique())
    num_countries = len(pd.read_csv(input_file)["NOC"].unique())

    medals_tensor = torch.zeros((num_unique_times, num_countries, 3))

    for row in pd.read_csv(input_file).iterrows():
        year = row[1]["Year"]
        country = row[1]["NOC"]
        medal = row[1]["Medal"]

        if medal == "Gold":
            medals_tensor[year][country][0] = 1
        elif medal == "Silver":
            medals_tensor[year][country][1] = 1
        elif medal == "Bronze":
            medals_tensor[year][country][2] = 1

def assign_id(input_file, output_file):
    # Load the data
    df = pd.read_csv(input_file)
    
    # Assign unique IDs to years
    unique_years = sorted(df['Year'].unique())
    year_to_id = {year: idx for idx, year in enumerate(unique_years)}
    df['Year'] = df['Year'].map(year_to_id)

    # Assign unique IDs to countries (NOC codes)
    unique_nocs = sorted(df['NOC'].unique())
    noc_to_id = {noc: idx for idx, noc in enumerate(unique_nocs)}
    df['NOC'] = df['NOC'].map(noc_to_id)

    # Assign unique IDs to sports
    unique_sports = sorted(df['Sport'].unique())
    sport_to_id = {sport: idx for idx, sport in enumerate(unique_sports)}
    df['Sport'] = df['Sport'].map(sport_to_id)

    # Assign unique IDs to events within each sport
    event_id_mapping = {}
    event_id_list = []

    for sport in df['Sport'].unique():
        sport_events = sorted(df[df['Sport'] == sport]['Event'].unique())
        event_mapping = {event: idx for idx, event in enumerate(sport_events)}
        event_id_mapping[sport] = event_mapping

        # Prepare data for mapping file
        for event, event_id in event_mapping.items():
            event_id_list.append({'Sport': sport, 'Event': event, 'Event_ID': event_id})

    df['Event'] = df.apply(lambda row: event_id_mapping[row['Sport']][row['Event']], axis=1)

    # Save the updated dataset with IDs replacing original values
    df.to_csv(output_file, index=False)

    # Create mapping dataframes
    noc_mapping_df = pd.DataFrame(list(noc_to_id.items()), columns=['NOC', 'NOC_ID'])
    sport_mapping_df = pd.DataFrame(list(sport_to_id.items()), columns=['Sport', 'Sport_ID'])
    # Make sure that the sports are sorted for the event file
    event_mapping_df = pd.DataFrame(event_id_list, columns=['Sport', 'Event', 'Event_ID'])
    event_mapping_df = event_mapping_df.sort_values(by=['Sport', 'Event_ID'])
    year_mapping_df = pd.DataFrame(list(year_to_id.items()), columns=['Year', 'Year_ID'])

    # Save mappings to CSV
    mappings_output_file = 'olympic_mappings.xlsx'
    with pd.ExcelWriter(mappings_output_file) as writer:
        year_mapping_df.to_excel(writer, sheet_name='Year_Mapping', index=False)
        noc_mapping_df.to_excel(writer, sheet_name='NOC_Mapping', index=False)
        sport_mapping_df.to_excel(writer, sheet_name='Sport_Mapping', index=False)
        event_mapping_df.to_excel(writer, sheet_name='Event_Mapping', index=False)

    print(f"Updated data file saved as: {output_file}")
    print(f"Mapping files saved as: {mappings_output_file}")


def combining_teams(input_file, output_file):
    # Load the data
    df = pd.read_csv(input_file)



    # Combine team sport entries based on year, NOC, sport, event, and teams column
    combined_team_sports = team_sports_df.groupby(['Year', 'NOC', 'Sport', 'Event', 'Team']).agg(aggregation_rules).reset_index()

    # Merge the individual and processed team sports data back together
    processed_df = pd.concat([individual_sports_df, combined_team_sports], ignore_index=True)

    # Save the processed dataset
    processed_df.to_csv(output_file, index=False)

    print(f"Processed data saved as: {output_file}")



#get_team_sports("raw_data/summerOly_athletes.csv", "summerOly_athletes_team.csv")
#get_irregularities("summerOly_athletes_team.csv")
#delete_irregulars("high_irregularities.txt", "summerOly_athletes_team.csv", "summerOly_athletes_team_no_high_irregulars.csv")
#map_ids("summerOly_athletes_team_no_high_irregulars.csv", "summerOly_athletes_team_id_no_high_irregulars.csv")
#assign_id("summerOly_athletes_team_id_no_high_irregulars.csv", "summerOly_athletes_team_id_no_high_irregulars_numerical.csv")
combining_teams("summerOly_athletes_team_id_no_high_irregulars_numerical.csv", "summerOly_athletes_team_id_no_high_irregulars_numerical_combined.csv")