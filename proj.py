"""
Authors: Chirayu Rai and Tony Montero
"""

from time import sleep as sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from imblearn.ensemble import BalancedRandomForestClassifier
from requests import get
from sklearn.model_selection import train_test_split
from sportsipy.nba.teams import Teams

# change this global var for where the csv is saved/read from
PATH_TO_CSV = './nba.csv'


def main():
    # Only uncomment this line if you do not already have data created
    #create_data_csv()
    cleaned_df = clean_csv_data()
    balance_dataset(cleaned_df)


def balance_dataset(df):
    labels = np.array(df.pop('Champion'))
    train, test, train_labels, test_labels = train_test_split(df,labels,stratify = labels, test_size= 0.25,random_state = 40) 
    model = BalancedRandomForestClassifier( n_estimators=100, random_state=40, max_features = 'sqrt',n_jobs=-1, verbose = 1)

    model.fit(train, train_labels)

    #Show model importances
    importances = model.feature_importances_*100
    indices = np.argsort(importances)[::-1]
    names = [train.columns[i] for i in indices]

    """ Doing this to see what features are important/hold the most weight when it comes to becoming a champion or not"""
    # Barplot: Add bars
    plt.bar(range(train.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(train.shape[1]), names, rotation=90, fontsize = 4)
    plt.yticks(range(0,15,5), fontsize=8)
    plt.grid(axis='x')
    # Create plot title
    plt.title("Feature Importances")
    # Show plot
    plt.show()


def clean_csv_data():
    # pull in data form csv (change this according to wherever you saved your data)
    df = pd.read_csv(PATH_TO_CSV)

    # Clean out the year and name column (year is duplicate and we have abv, so no need for name)
    df.drop('Year', axis=1, inplace=True)
    df.drop('name', axis=1, inplace=True)

    # If champion --> Yes, if not champion --> No
    df['Champion'] = df["Champion"].replace('.*', 1, regex=True)
    df['Champion'] = df["Champion"].fillna(0)

    # Creating an id for each team within each season to avoid collisions
    #df['id'] = f"{df['abbreviation']}-{df['season'].astype('int').astype('str')}"

    # don't need abv
    df = df.drop('abbreviation', axis=1)

    # Scaling all data to account for games played, because it's not necessary that every season has 82 games played
    cols = df.columns.to_list()
    columns_to_not_scale = ['season', 'Champion', 'games_played', 'rank'] + [col for col in cols if 'percentage' in col]
    columns_to_scale = [col for col in cols if col not in columns_to_not_scale]
    scaled_df = df[columns_to_scale].div(df['games_played'], axis=0)
    df = pd.concat([df[columns_to_not_scale], scaled_df], axis=1)

    return df


def create_data_csv():
    """
    We are pulling data from 1980 and on, because that's when the three point line was first implemented
    This will pull a huge set of data and then save it as a csv file.
    """
    teams_stats_list = [] 
    # Iterate through all teams from 1980 - 2023
    for year in range(1980, 2023):
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
