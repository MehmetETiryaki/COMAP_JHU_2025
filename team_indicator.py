import pandas as pd

team_keyword_list = set(["Relay", "Double", "Team", "Mixed", "Two Person", "Three Person", "Four Person", "Five Person", "Six Person", "Pairs", "Fours", "Quadruple", "Sixes", "Eights",
                         "Basketball", "Baseball", "Volleyball", "Tandem", "Football", "Handball", "Hockey", "Curling", "Water Polo", "Rugby", "Rowing", "Soccer", "Softball", "Rugby", "Polo", "Cricket", "Lacrosse", "Tug-Of-War"])

input_file = "2025_Problem_C_Data/summerOly_athletes.csv"
output_file = "2025_Problem_C_Data/summerOly_athletes_team.csv"
df = pd.read_csv(input_file)
# create a column named Team_Sport that is 1 if either column "Sport" or "Event" contains a team keyword in team_keyword_list, 0 otherwise
df["Team_Sport"] = df.apply(lambda row: 1 if any(keyword in row["Sport"] for keyword in team_keyword_list) or any(keyword in row["Event"] for keyword in team_keyword_list) else 0, axis=1)
df.to_csv(output_file, index=False)