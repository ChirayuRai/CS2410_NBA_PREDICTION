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
useless_features = ['two_point_field_goals', 'opp_points', 'opp_steals', 'opp_field_goal_attempts', 'opp_two_point_field_goal_percentage', 'defensive_rebounds', 'turnovers', 'games_played', 'opp_three_point_field_goal_attempts', 'free_throws', 'steals', 'free_throw_percentage', 'personal_fouls', 'opp_personal_fouls', 'offensive_rebounds', 'minutes_played', 'opp_three_point_field_goal_percentage', 'two_point_field_goal_attempts', 'opp_assists', 'total_rebounds', 'free_throw_attempts', 'opp_defensive_rebounds']


def main():
    # Only uncomment this line if you do not already have data created
    #create_historical_data_csv()
    # create_curr_year_csv()
    finish_model()


def finish_model():
    df = pd.read_csv("./nba.csv")
    df.drop(index=df.index[-1],axis=0,inplace=True)

    # Split data into features and target variable
    y = np.array(df.pop("Champion"))
    X = df
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=40)

    # Dropping columns we dont need
    X_train.drop(["name", "season"], axis=1, inplace=True)
    X_test.drop(["name", "season"], axis=1, inplace=True)

    # Create the model while also balancing the dataset
    balanced = BalancedRandomForestClassifier(random_state=20, n_estimators=200, warm_start=True, n_jobs=-1)
    balanced.fit(X_train, y_train)

    # Using the importances code down below, I printed out the least important features, and decided to drop 
    # the less useful ones from the dataset to increase the overall accuracy 
    # can see it being dropped in the creation of each csv
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
    current_year_data = pd.read_csv(f"./{CURRENT_YEAR}_data.csv")

    # popping to add later
    curr_name = current_year_data.pop("name")

    probabilities = balanced.predict_proba(current_year_data)[:, 1]

    # adding names back to probabilities
    final_model = pd.concat([pd.DataFrame(probabilities), curr_name, current_year_data], axis=1)
    final_model=final_model.rename(columns={0:'probabilities'})

    final_model.sort_values(by='probabilities', ascending=False, inplace=True)
    final_model.to_csv("./final.csv")

    print()
    print("--- FINAL PROBABILITY LIST ---")
    print(final_model["name"])


def create_curr_year_csv():
    teams_stat_list = []
    for team in Teams(CURRENT_YEAR):
        team_df = team.dataframe
        teams_stat_list.append(team_df)

    # fixing up data and saving to disk
    df = pd.concat(teams_stat_list).reset_index(drop=True)
    df.drop(useless_features, axis=1, inplace=True)
    df.drop(["abbreviation"], axis=1, inplace=True)
    df.to_csv(f"./{CURRENT_YEAR}_data.csv", index=False)

    df = df.reindex(sorted(df.columns), axis=1)
    return df


def create_historical_data_csv():
    """
    We are pulling data from 1980 and on, because that's when the three point line was first implemented
    This will pull a huge set of data and then save it as a csv file.
    """
    teams_stats_list = [] 
    # Iterate through all teams from 1980 - 2022
    for year in range(1980, CURRENT_YEAR):
        # Iterate through every team that season
        for team in Teams(year):
            sleep(0.15)
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

    # If champion --> Yes, if not champion --> No
    final_df['Champion'] = final_df["Champion"].replace('.*', 1, regex=True)
    final_df['Champion'] = final_df["Champion"].fillna(0)
    

    
    final_df.drop(["Year", "abbreviation"], axis=1, inplace=True)
    # Scaling all data to account for games played, because it's not necessary that every season has 82 games played
    cols = final_df.columns.to_list()
    columns_to_not_scale = ['season', 'Champion', 'games_played', 'rank', 'name'] + [col for col in cols if 'percentage' in col]
    columns_to_scale = [col for col in cols if col not in columns_to_not_scale]
    scaled_df = final_df[columns_to_scale].div(final_df['games_played'], axis=0)
    final_df = pd.concat([final_df[columns_to_not_scale], scaled_df], axis=1)
    final_df = final_df.reindex(sorted(final_df.columns), axis=1)

    # Cleaning out useless features for increased accuracy
    final_df.drop(useless_features, axis=1, inplace=True)
    
    final_df.to_csv(PATH_TO_CSV, index=False)


if __name__ == "__main__":
    main()
