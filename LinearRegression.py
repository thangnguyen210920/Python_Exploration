import time
import os

import pandas as pd # for dataframe operations.
import numpy as np #for linear algebra operations.
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt # for plotting

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy.stats import kendalltau # we will compare predicted and real results

points_df = pd.read_excel('Data\\TurkeySuperLeague.xlsx', sheet_name='Points')
players_df = pd.read_excel('Data\\TurkeySuperLeague.xlsx', sheet_name='Player')

# Adding ranking column to the points table
points_df['Ranking'] = points_df.groupby('Season')['Points'].rank(ascending=False,method='first')

points_group_sc = points_df.set_index(['Season','Club']).sort_index()
players_group_sc = players_df.set_index(['Season','Club']).sort_index()

# Joining the dataframes together as a new dataframe
whole_df = pd.DataFrame(players_group_sc)
whole_df['Points'] = points_group_sc['Points'].copy()
whole_df.head()
# let's get some description about our main dataframe
whole_df.describe()

correlation = whole_df.corr() #corr() method of pandas library calculates correlation between columns of dataframe

sns.heatmap(correlation,cmap="YlGnBu",annot=True)
plt.show()

whole_df['Ranking'] = points_group_sc['Ranking'].copy()
data_means = players_df.groupby(['Season', 'Club']).mean()[['Market Value', 'Age','Foreign','Multinational']]
data_sums = players_df.groupby(['Season', 'Club']).sum()[['Market Value','Foreign','Multinational']]
total_player= players_df.groupby(['Season', 'Club']).count()['Player']
data_standart_deviations= players_df.groupby(['Season', 'Club']).std()[['Market Value', 'Age']]
feature_df = pd.DataFrame(data_means)
player_counts_per_season = players_df.groupby(['Season', 'Club']).count()[['Player']]
feature_df['Value Ranking'] = data_means.groupby('Season')['Market Value'].rank(ascending=False,method='first')

main_df = pd.DataFrame(data_means)
#main_df['Value Ranking'] = data_means.groupby('Season')['Market Value'].rank(ascending=False,method='first')

main_df['Points'] = points_group_sc['Points'].copy()
main_df['Ranking'] = points_group_sc['Ranking'].copy()
main_df['age_std'] = data_standart_deviations['Age'].copy()
main_df['multi_sum'] = data_sums['Multinational'].copy()
main_df['foreign_sum'] = data_sums['Foreign'].copy()
main_df['player_sum'] = total_player
main_df['foreign_ratio'] = (main_df['foreign_sum'] / main_df['player_sum']).copy()
main_df['market_sum'] = data_sums['Market Value'].copy()
main_df['market_std'] = data_standart_deviations['Market Value'].copy()

normalized_means = data_means.groupby(['Season']).transform(lambda x: x/x.mean())
main_df['market_norm'] = normalized_means['Market Value']
main_df['age_norm'] = normalized_means['Age']
main_df.head()

features=main_df.columns

for i in features:
    sns.lmplot(x=i, y="Points", data=main_df,line_kws={'color': 'red'},size=4)
    text="Relation between Points and " + i
    plt.title(text)
    plt.show()

correlation = main_df.corr()['Points']
# convert series to dataframe so it can be sorted
correlation_df = pd.DataFrame(correlation)
# correct column label from Points to correlation
correlation_df.columns = ["Correlation"]
# sort correlation
corr_sorted = correlation_df.sort_values(by=['Correlation'], ascending=False)
corr_sorted.head(40)

def convert_points_to_predictions(predictions, shouldAscend):
    predictions['Real Rank'] = predictions['Real'].rank(ascending=shouldAscend,method='first')
    predictions['Predicted Rank'] = predictions['Predict'].rank(ascending=shouldAscend,method='first')
    return predictions


def train_test_on_points(final_df):
    tau_ = 0
    for season in range(2007, 2015):
        train = final_df[final_df['Season'] != season]  # train data is not contain target season
        test = final_df[final_df['Season'] == season]  # test data is all about target season.
        # We will try to predict Points.So,we have to drop it from the test dataframe.
        # Additionaly, we don't need Season information anymore. Let's drop both
        X_train = train.drop(['Points', 'Season'], axis=1)
        # we don't have to normalize y_train data; but it is proved normalization has a positive effect on accuracy.

        y_train = train['Points'].transform(lambda x: (x - x.mean()) / x.std())
        # we should also drop Points and Season columns from X_test
        X_test = test.drop(['Points', 'Season'], axis=1)
        # normalize y_test
        y_test = test['Points'].transform(lambda x: (x - x.mean()) / x.std())

        # X_train, X_test, y_train , y_test = train_test_split(final_df.drop('Points',axis=1),final_df['Points'],
        #                                                    test_size=0.1, random_state=1)
        # This is the regression model you will use
        final_model = LinearRegression(fit_intercept=False)
        # Fit our X_train and y_train for learning
        final_model.fit(X_train, y_train)
        # get scores
        score = cross_val_score(final_model, X_train, y_train, cv=10)
        print(score)

        y_predict = final_model.predict(X_test)

        preds = pd.DataFrame({"Predict": y_predict})

        preds['Real'] = y_test.reset_index().iloc[:, -1]

        ranks = pd.DataFrame()

        ranks['Real Rank'] = preds['Real'].rank(ascending=False, method='first')

        ranks['Predicted Rank'] = preds['Predict'].rank(ascending=False, method='first')
        tau, _ = kendalltau(ranks['Predicted Rank'], ranks['Real Rank'])
        tau_ += tau

    print('\n')
    print('kendalltau for Points estimation: ', tau_ / 9)


final_df = main_df.copy()
final_df.dropna(inplace=True)
final_df = final_df.reset_index()
get_dummy_for_clup_names = pd.get_dummies(final_df['Club'], drop_first=True)
final_df = final_df.join(get_dummy_for_clup_names)
final_df.drop(columns=['Club'], inplace=True)

train_test_on_points(final_df)

season = 2015
# to predict the ranking of 2015
final_df=main_df.copy()
final_df.dropna(inplace=True)
final_df = final_df.reset_index()
get_dummy_for_clup_names=pd.get_dummies(final_df['Club'],drop_first=True)
final_df=final_df.join(get_dummy_for_clup_names)
final_df.drop(columns=['Club'],inplace=True)

train = final_df[final_df['Season']!=season]
test = final_df[final_df['Season']==season]
X_train = train.drop(['Ranking','Season'],axis=1)
y_train = train['Ranking']
X_test = test.drop(['Ranking','Season'],axis=1)
y_test = test['Ranking']
# This is the regression model you will use
final_model = LinearRegression(fit_intercept=False)
final_model.fit(X_train,y_train)
score = cross_val_score(final_model,X_train, y_train,cv=10)
# print(score)
y_predict = final_model.predict(X_test)
#a = mean_squared_error(y_test,y_predict)
preds_rank = pd.DataFrame({"Predict":y_predict})
preds_rank['Real']= y_test.reset_index().iloc[:,-1]
# ranks.head()
# show ranks
print(preds_rank.sort_values(by='Real',ascending=True))
tau, _ = kendalltau(preds_rank['Predict'], preds_rank['Real'])
# Print tau both to file and screen
print('\n')
print('kendalltau for rank estimation:',tau)
