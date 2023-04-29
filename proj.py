"""
Authors: Chirayu Rai and Tony Montero
"""

""" suppressing useless warnings from SKLearn """
def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn


import datetime
from time import sleep as sleep

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from imblearn.ensemble import BalancedRandomForestClassifier
from requests import get
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sportsipy.nba.teams import Teams

# change this global var for where the csv is saved/read from
PATH_TO_CSV = './nba.csv'
CURRENT_YEAR = int(datetime.date.today().year)
LAST_YEAR = CURRENT_YEAR - 1


def main():
    # Only uncomment this line if you do not already have data created
    #create_data_csv()
    cleaned_df = clean_csv_data()
    finish_model(cleaned_df)


def finish_model(df):
    # Using the importances code down below, I printed out the least important features, and decided to drop 
    # the less useful ones from the dataset to increase the overall accuracy 
    train_drop = ['two_point_field_goals', 'opp_points', 'opp_steals', 'opp_field_goal_attempts', 'opp_two_point_field_goal_percentage', 'defensive_rebounds', 'turnovers', 'games_played', 'opp_three_point_field_goal_attempts', 'free_throws', 'steals', 'free_throw_percentage', 'personal_fouls', 'opp_personal_fouls', 'offensive_rebounds', 'minutes_played', 'opp_three_point_field_goal_percentage', 'two_point_field_goal_attempts', 'opp_assists', 'total_rebounds', 'free_throw_attempts', 'opp_defensive_rebounds']
    # Split data into features and target variable
    y = np.array(df.pop("Champion"))
    X = df.drop(train_drop, axis=1)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=40)

    # Dropping columns we dont need
    X_train.drop(["name", "season"], axis=1, inplace=True)
    X_test.drop(["name", "season"], axis=1, inplace=True)

    # Create the model while also balancing the dataset
    balanced = BalancedRandomForestClassifier(random_state=20, n_estimators=200, warm_start=True, n_jobs=-1)
    balanced.fit(X_train, y_train)

    """ Where we start finding features to cut out """
    importances = balanced.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_train.columns[i] for i in indices]

    importances_df = pd.DataFrame({'features': names, 'importance': importances})
    importances_df.sort_values(by=['importance'], ascending=False, axis=0, inplace=True)
    feature_arr = np.array(importances_df.pop("features"))
    feature_arr = list(feature_arr)

    # Printing out the features that aren't the 20 most important ones so we can drop them and increase accuracy
    #print([feature_arr[i] for i in range(20, len(feature_arr))])
    
    # starting predictions in order to evaluate model
    train_pred = balanced.predict(X_train)
    test_pred = balanced.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    roc_auc = roc_auc_score(y_test, test_pred)

    print("--- METRICS ---")
    print("Accuracy:", accuracy)
    print("F1-score:", f1)
    print("AUC-ROC:", roc_auc)

    # Assuming you have collected the data for the current year's teams
    # If data is not already created, invoke create_2023_df()
    current_year_data = pd.read_csv("./2023.csv")

    # popping to add later
    curr_name = current_year_data.pop("name")
    # removing useless data
    current_year_data.drop(["abbreviation", "season"], axis=1, inplace=True)
    current_year_data.drop(train_drop, axis=1, inplace=True)

    probabilities = balanced.predict_proba(current_year_data)[:, 1]

    # adding names back to probabilities
    final_model = pd.concat([pd.DataFrame(probabilities), curr_name], axis=1)
    final_model=final_model.rename(columns={0:'probabilities'})

    final_model.sort_values(by='probabilities', ascending=False, inplace=True)
    final_model.to_csv("./final.csv")

    print()
    print("--- FINAL PROBABILITY LIST ---")
    print(final_model["name"])


def clean_csv_data():
    # pull in data form csv (change this according to wherever you saved your data)
    df = pd.read_csv(PATH_TO_CSV)

    # Clean out the year and name column (year is duplicate and we have abv, so no need for name)
    df.drop(['Year', "abbreviation"], axis=1, inplace=True)

    # If champion --> Yes, if not champion --> No
    df['Champion'] = df["Champion"].replace('.*', 1, regex=True)
    df['Champion'] = df["Champion"].fillna(0)
    

    # Scaling all data to account for games played, because it's not necessary that every season has 82 games played
    cols = df.columns.to_list()
    columns_to_not_scale = ['season', 'Champion', 'games_played', 'rank', 'name'] + [col for col in cols if 'percentage' in col]
    columns_to_scale = [col for col in cols if col not in columns_to_not_scale]
    scaled_df = df[columns_to_scale].div(df['games_played'], axis=0)
    df = pd.concat([df[columns_to_not_scale], scaled_df], axis=1)

    df = df.reindex(sorted(df.columns), axis=1)
    return df

def create_curr_year_df():
    teams_stat_list = []
    for team in Teams(CURRENT_YEAR):
        team_df = team.dataframe
        team_df['season'] = CURRENT_YEAR 
        teams_stat_list.append(team_df)

    df = pd.concat(teams_stat_list).reset_index(drop=True)
    df.to_csv(f"./{CURRENT_YEAR}.csv", index=False)


    # don't need abv
    df = df.drop('abbreviation', axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def create_data_csv():
    """
    We are pulling data from 1980 and on, because that's when the three point line was first implemented
    This will pull a huge set of data and then save it as a csv file.
    """
    teams_stats_list = [] 
    # Iterate through all teams from 1980 - 2022
    for year in range(1980, CURRENT_YEAR):
        # Iterate through every team that season
        sleep(0.1)
        for team in Teams(year):
            sleep(0.2)
            team_df = team.dataframe
            team_df["season"] = year
            teams_stats_list.append(team_df)
    teams_df = pd.concat(teams_stats_list).reset_index(drop=True)

    # scraping basketball reference for the champion of every year referenced
    r = get(f"https://www.basketball-reference.com/playoffs/")
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, 'html.parser')
        table = soup.find('table', attrs={'id': 'champions_index'})
        if table:
            # Basically just filtering results down to champion and year,
            # then restructuring in order to have year = champion or not
            champ_df = pd.read_html(str(table))[0]
            champ_df = champ_df.droplevel(level=0, axis=1)
            champ_df = champ_df[champ_df['Lg'] == 'NBA']
            champ_df = champ_df[champ_df['Year'] > 1979]
            champ_df = champ_df[['Year', 'Champion']]
            champ_df["Year"] = champ_df["Year"].astype(int)

    # Merge the champion and team stats
    final_df = pd.merge(teams_df, champ_df, left_on=['name', 'season'], right_on=['Champion', 'Year'], how='outer')

    # cleaning up abbreviation keys
    final_df = final_df[final_df['abbreviation'].notna()]
    
    final_df.to_csv(PATH_TO_CSV, index=False)


if __name__ == "__main__":
    main()
