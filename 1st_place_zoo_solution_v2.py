"""
Converted from 1st_place_zoo_solution_v2.ipynb.
This script keeps notebook code in execution order and adds markdown context as comments.
"""


# ===== Markdown cell 00 =====
# # NFL Big Data Bowl 2020 - 1st place solution The Zoo


# ===== Markdown cell 01 =====
# This notebook aims to reproduce the NFL Big Data Bowl 2020 winner solution described in [1]. The purpose of the competiton was to develop a model to predict how many yards a team will gain on given rushing plays as they happen [2]. The dataset contains game, play, and player-level data. This elegant solution is only based on player-level data. In particular, on relative location and speed features only.
# 
# To understand the proposed solution, assume that in a simplified definition, a rushing play consists on:
# - A rusher, whose aim is to run forward as far as possible
# - 11 defense players who are trying to stop the rusher
# - 10 remaining offense players trying to prevent defenders from blocking or tackling the rusher
# 
# Based on this simplified version of the game, the authors in [1] came up with the following network structure:
# 
# <img src="images/model_structure.png" style="width:680px;height:200px;">
# 
# 
# We will go into the details throughout this notebook.
# 
# The remainder of this notebook is organized as follows. Section 1 describes and contains the code for data processing and data augmentation. Section 2 provides the model structure. Finally, section 3 draws some conclusions and some possible improvements


# ===== Markdown cell 02 =====
# ## Data Processing


# ===== Code cell 03 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import utils


# ===== Code cell 04 =====
train = pd.read_csv('data/train.csv', dtype={'WindSpeed': 'object'})


# ===== Markdown cell 05 =====
# First, we divide the dataset into two daframes. The first dataframe (df_players) contains the columns related to the player-level features. Meanwhile, the second dataframe (df_play) is formed by some play-level features which will be useful to perform some transformations on df_players.


# ===== Code cell 06 =====
def split_play_and_player_cols(df, predicting=False):
    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    
    df['PlayId'] = df['PlayId'].astype(str)
    
    # We must assume here that the first 22 rows correspond to the same player:
    player_cols = [
        'PlayId', # This is the link between them
        'Season',
        'Team',
        'X',
        'Y',
        'S',
        'Dis',
        'Dir',
        'NflId',
        'IsRusher',
    ]

    df_players = df[player_cols]
    
    play_cols = [
        'PlayId',
        'Season',
        'PossessionTeam',
        'HomeTeamAbbr',
        'VisitorTeamAbbr',
        'PlayDirection', 
        'FieldPosition',
        'YardLine',
    ]
    
    if not predicting:
        play_cols.append('Yards')
        
    df_play = df[play_cols].copy()

    ## Fillna in FieldPosition attribute
    #df['FieldPosition'] = df.groupby(['PlayId'], sort=False)['FieldPosition'].apply(lambda x: x.ffill().bfill())
    
    # Get first 
    df_play = df_play.groupby('PlayId').first().reset_index()

    print('rows/plays in df: ', len(df_play))
    assert df_play.PlayId.nunique() == df.PlayId.nunique(), "Play/player split failed?"  # Boom
    
    return df_play, df_players

play_ids = train["PlayId"].unique()

df_play, df_players = split_play_and_player_cols(train)


# ===== Markdown cell 07 =====
# We have some problems with the enconding of the teams such as BLT and BAL or ARZ and ARI. Let's fix it.


# ===== Code cell 08 =====
def process_team_abbr(df):

    #These are only problems:
    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
    for abb in train['PossessionTeam'].unique():
        map_abbr[abb] = abb

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

    df['HomePossession'] = df['PossessionTeam'] == df['HomeTeamAbbr']
    
    return

process_team_abbr(df_play)


# ===== Code cell 09 =====
def process_play_direction(df):
    df['IsPlayLeftToRight'] = df['PlayDirection'].apply(lambda val: True if val.strip() == 'right' else False)
    return

process_play_direction(df_play)


# ===== Markdown cell 10 =====
# We compute how many yards are left to the end-zone.


# ===== Code cell 11 =====
def process_yard_til_end_zone(df):
    def convert_to_yardline100(row):
        return (100 - row['YardLine']) if (row['PossessionTeam'] == row['FieldPosition']) else row['YardLine']
    df['Yardline100'] = df.apply(convert_to_yardline100, axis=1)
    return

process_yard_til_end_zone(df_play)


# ===== Markdown cell 12 =====
# Now, we add the computed features to df_players


# ===== Code cell 13 =====
df_players = df_players.merge(
    df_play[['PlayId', 'PossessionTeam', 'HomeTeamAbbr', 'PlayDirection', 'Yardline100']], 
    how='left', on='PlayId')


# ===== Code cell 14 =====
df_players.loc[df_players.Season == 2017].plot.scatter(x='Dis', y='S', title='Season 2017',grid=True)


# ===== Code cell 15 =====
df_players.loc[df_players.Season == 2018].plot.scatter(x='Dis', y='S', title='Season 2018', grid=True)


# ===== Markdown cell 16 =====
# In 2018 data we can see that S is linearly related to Dis. However, data in 2017 is not very fit. Using a linear regresion to fit the 2018 data, we found that S can be replaced by 10*Dir. This give an improvment in the predictions


# ===== Code cell 17 =====
X = df_players.loc[df_players.Season == 2018]['Dis']
y = df_players.loc[df_players.Season == 2018]['S']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit() 
model.summary()


# ===== Code cell 18 =====
df_players.loc[df_players.Season == 2017, 'S'] = 10*df_players.loc[df_players.Season == 2017,'Dis']


# ===== Markdown cell 19 =====
# Now, let's adjusted the data to always be from left to right.


# ===== Code cell 20 =====
def standarize_direction(df):
    # adjusted the data to always be from left to right
    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']

    df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi/180.0

    df['ToLeft'] = df.PlayDirection == "left"
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['IsOnOffense'] = df.Team == df.TeamOnOffense # Is player on offense?
    df['X_std'] = df.X
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X']
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y_std'] = 160/3 - df.loc[df.ToLeft, 'Y']
    df['Dir_std'] = df.Dir_rad
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2*np.pi)
   
    #Replace Null in Dir_rad
    df.loc[(df.IsOnOffense) & df['Dir_std'].isna(),'Dir_std'] = 0.0
    df.loc[~(df.IsOnOffense) & df['Dir_std'].isna(),'Dir_std'] = np.pi

standarize_direction(df_players)


# ===== Markdown cell 21 =====
# We adjust only the plays moving to left. To explain the transformation, consider the following images that show two original plays (the purple is the team in offense)


# ===== Code cell 22 =====
for play_id in ['20170910001102', '20170910000081']: 
    utils.show_play(play_id, df_players)


# ===== Markdown cell 23 =====
# Now, these are the same plays after the transformation


# ===== Code cell 24 =====
for play_id in ['20170910001102', '20170910000081']: 
    utils.show_play_std(play_id, df_players)


# ===== Markdown cell 25 =====
# Note that we only modify the plays moving to left. The source code to these plots is taken from [3]


# ===== Markdown cell 26 =====
# ### Data augmentation
# For training, we assume that in a mirrored world the runs would have had the same outcomes. We apply 50% augmentation to flip the Y coordinates (and all respective relative features emerging from it). Furthermore, the function process_tracking_data computes the projections on X and Y for the velocity of each player and other features relative to rusher.


# ===== Code cell 27 =====
def data_augmentation(df, sample_ids):
    df_sample = df.loc[df.PlayId.isin(sample_ids)].copy()
    df_sample['Y_std'] = 160/3  - df_sample['Y_std']
    df_sample['Dir_std'] = df_sample['Dir_std'].apply(lambda x: 2*np.pi - x)
    df_sample['PlayId'] = df_sample['PlayId'].apply(lambda x: x+'_aug')
    return df_sample

def process_tracking_data(df):
    # More feature engineering for all:
    df['Sx'] = df['S']*df['Dir_std'].apply(math.cos)
    df['Sy'] = df['S']*df['Dir_std'].apply(math.sin)
    
    # ball carrier position
    rushers = df[df['IsRusher']].copy()
    rushers.set_index('PlayId', inplace=True, drop=True)
    playId_rusher_map = rushers[['X_std', 'Y_std', 'Sx', 'Sy']].to_dict(orient='index')
    rusher_x = df['PlayId'].apply(lambda val: playId_rusher_map[val]['X_std'])
    rusher_y = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Y_std'])
    rusher_Sx = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Sx'])
    rusher_Sy = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Sy'])
    
    # Calculate differences between the rusher and the players:
    df['player_minus_rusher_x'] = rusher_x - df['X_std']
    df['player_minus_rusher_y'] = rusher_y - df['Y_std']

    # Velocity parallel to direction of rusher:
    df['player_minus_rusher_Sx'] = rusher_Sx - df['Sx']
    df['player_minus_rusher_Sy'] = rusher_Sy - df['Sy']

    return

sample_ids = np.random.choice(df_play.PlayId.unique(), int(0.5*len(df_play.PlayId.unique())))

df_players_aug = data_augmentation(df_players, sample_ids)
df_players = pd.concat([df_players, df_players_aug])
df_players.reset_index()

df_play_aug = df_play.loc[df_play.PlayId.isin(sample_ids)].copy()
df_play_aug['PlayId'] = df_play_aug['PlayId'].apply(lambda x: x+'_aug')
df_play = pd.concat([df_play, df_play_aug])
df_play.reset_index()

# This is necessary to maintain the order when in the next cell we use groupby
df_players.sort_values(by=['PlayId'],inplace=True)
df_play.sort_values(by=['PlayId'],inplace=True)

process_tracking_data(df_players)


# ===== Code cell 28 =====
tracking_level_features = [
    'PlayId',
    'IsOnOffense',
    'X_std',
    'Y_std',
    'Sx',
    'Sy',
    'player_minus_rusher_x',
    'player_minus_rusher_y',
    'player_minus_rusher_Sx',
    'player_minus_rusher_Sy',
    'IsRusher'
]

df_all_feats = df_players[tracking_level_features]

print('Any null values: ', df_all_feats.isnull().sum().sum())


# ===== Markdown cell 29 =====
# Finally, we create the train tensor to feed the convolutional network. The following image depicts the structure of the input tensor:
# 
# <img src="images/input.png" style="width:350px;height:160px;">
# 
# Note that the idea is to reshape the data of a play into a tensor of defense vs offense, using features as channels to apply 2d operations (The figure does not follow the convention on ConvNet, and the channels are in the z-axis). There are 5 vector features which were important (so 10 numeric features if you count projections on X and Y axis). The vectors are relative locations and speeds, so to derive them we used only ‘X’, ‘Y’, ‘S’ and ‘Dir’ variables from data.


# ===== Code cell 30 =====
# Notebook magic removed: %%time

grouped = df_all_feats.groupby('PlayId')
train_x = np.zeros([len(grouped.size()),11,10,10])
i = 0
play_ids = df_play.PlayId.values
for name, group in grouped:
    if name!=play_ids[i]:
        print("Error")

    [[rusher_x, rusher_y, rusher_Sx, rusher_Sy]] = group.loc[group.IsRusher==1,['X_std', 'Y_std','Sx','Sy']].values

    offense_ids = group[group.IsOnOffense & ~group.IsRusher].index
    defense_ids = group[~group.IsOnOffense].index

    for j, defense_id in enumerate(defense_ids):
        [def_x, def_y, def_Sx, def_Sy] = group.loc[defense_id,['X_std', 'Y_std','Sx','Sy']].values
        [def_rusher_x, def_rusher_y] = group.loc[defense_id,['player_minus_rusher_x', 'player_minus_rusher_y']].values
        [def_rusher_Sx, def_rusher_Sy] =  group.loc[defense_id,['player_minus_rusher_Sx', 'player_minus_rusher_Sy']].values
        
        train_x[i,j,:,:4] = group.loc[offense_ids,['Sx','Sy','X_std', 'Y_std']].values - np.array([def_Sx, def_Sy, def_x,def_y])
        train_x[i,j,:,-6:] = [def_rusher_Sx, def_rusher_Sy, def_rusher_x, def_rusher_y, def_Sx, def_Sy]
    
    i+=1

np.save('data/train_x_v3(augmented-50).npy', train_x)


# ===== Markdown cell 31 =====
# Additionally, for training we clip the target to -30 and 50. 


# ===== Code cell 32 =====
# Transform Y into indexed-classes:
train_y = df_play[['PlayId', 'Yards']].copy()

train_y['YardIndex'] = train_y['Yards'].apply(lambda val: val + 99)

min_idx_y = 71
max_idx_y = 150

train_y['YardIndexClipped'] = train_y['YardIndex'].apply(
    lambda val: min_idx_y if val < min_idx_y else max_idx_y if val > max_idx_y else val)

print('max yardIndex: ', train_y.YardIndex.max())
print('max yardIndexClipped: ', train_y.YardIndexClipped.max())
print('min yardIndex: ', train_y.YardIndex.min())
print('min yardIndexClipped: ', train_y.YardIndexClipped.min())

train_y.to_pickle('data/train_y_v3.pkl')


# ===== Code cell 33 =====
df_season = df_play[['PlayId', 'Season']].copy()
df_season.to_pickle('data/df_season_v3.pkl')


# ===== Markdown cell 34 =====
# ## Train ConvNet


# ===== Code cell 35 =====
train_x = np.load('data/train_x_v3(augmented-50).npy') 
train_y = pd.read_pickle('data/train_y_v3.pkl') 
df_season = pd.read_pickle('data/df_season_v3.pkl')

#num_classes_y = 199
min_idx_y = 71
max_idx_y = 150
num_classes_y = max_idx_y - min_idx_y + 1


# ===== Code cell 36 =====
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AvgPool1D, AvgPool2D, Reshape,
    Input, Activation, BatchNormalization, Dense, Add, Lambda, Dropout, LayerNormalization)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping

import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def crps(y_true, y_pred):
    loss = K.mean(K.sum((K.cumsum(y_pred, axis = 1) - K.cumsum(y_true, axis=1))**2, axis=1))/199
    return loss


# ===== Markdown cell 37 =====
# Let's define the newtork arquitecture. The simplified NN structure looks like this:
# 
# <img src="images/model_structure.png" style="width:680px;height:200px;">
# 
# "So the first block of convolutions learns to work with defense-offense pairs of players, using geometric features relative to rusher. The combination of multiple layers and activations before pooling was important to capture the trends properly. The second block of convolutions learns the necessary information per defense player before the aggregation. And the third block simply consists of dense layers and the usual things around them. 3 out of 5 input vectors do not depend on the offense player, hence they are constant across “off” dimension of the tensor." [1]


# ===== Code cell 38 =====
def get_conv_net(num_classes_y):
    #_, x, y, z = train_x.shape
    inputdense_players = Input(shape=(11,10,10), name = "playersfeatures_input")
    
    X = Conv2D(128, kernel_size=(1,1), strides=(1,1), activation='relu')(inputdense_players)
    X = Conv2D(160, kernel_size=(1,1), strides=(1,1), activation='relu')(X)
    X = Conv2D(128, kernel_size=(1,1), strides=(1,1), activation='relu')(X)
    
    # The second block of convolutions learns the necessary information per defense player before the aggregation.
    # For this reason the pool_size should be (1, 10). If you want to learn per off player the pool_size must be 
    # (11, 1)
    Xmax = MaxPooling2D(pool_size=(1,10))(X)
    Xmax = Lambda(lambda x1 : x1*0.3)(Xmax)

    Xavg = AvgPool2D(pool_size=(1,10))(X)
    Xavg = Lambda(lambda x1 : x1*0.7)(Xavg)

    X = Add()([Xmax, Xavg])
    X = Lambda(lambda y : K.squeeze(y,2))(X)
    X = BatchNormalization()(X)
    
    X = Conv1D(160, kernel_size=1, strides=1, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv1D(96, kernel_size=1, strides=1, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv1D(96, kernel_size=1, strides=1, activation='relu')(X)
    X = BatchNormalization()(X)
    
    Xmax = MaxPooling1D(pool_size=11)(X)
    Xmax = Lambda(lambda x1 : x1*0.3)(Xmax)

    Xavg = AvgPool1D(pool_size=11)(X)
    Xavg = Lambda(lambda x1 : x1*0.7)(Xavg)

    X = Add()([Xmax, Xavg])
    X = Lambda(lambda y : K.squeeze(y,1))(X)
    
    X = Dense(96, activation="relu")(X)
    X = BatchNormalization()(X)

    X = Dense(256, activation="relu")(X)
    X = LayerNormalization()(X)
    X = Dropout(0.3)(X)

    outsoft = Dense(num_classes_y, activation='softmax', name = "output")(X)

    model = Model(inputs = [inputdense_players], outputs = outsoft)
    return model


# ===== Code cell 39 =====
class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_valid, y_valid = self.data[0], self.data[1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
        logs['val_CRPS'] = val_s
        
        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


# ===== Code cell 40 =====
# Notebook magic removed: %%time

models = []
kf = KFold(n_splits=8, shuffle=True, random_state=42)
score = []

for i, (tdx, vdx) in enumerate(kf.split(train_x, train_y)):
    print(f'Fold : {i}')
    X_train, X_val = train_x[tdx], train_x[vdx],
    y_train, y_val = train_y.iloc[tdx]['YardIndexClipped'].values, train_y.iloc[vdx]['YardIndexClipped'].values
    season_val = df_season.iloc[vdx]['Season'].values

    y_train_values = np.zeros((len(y_train), num_classes_y), np.int32)
    for irow, row in enumerate(y_train):
        y_train_values[(irow, row - min_idx_y)] = 1
        
    y_val_values = np.zeros((len(y_val), num_classes_y), np.int32)
    for irow, row in enumerate(y_val - min_idx_y):
        y_val_values[(irow, row)] = 1

    val_idx = np.where(season_val!=2017)
    
    X_val = X_val[val_idx]
    y_val_values = y_val_values[val_idx]

    y_train_values = y_train_values.astype('float32')
    y_val_values = y_val_values.astype('float32')
    
    model = get_conv_net(num_classes_y)

    es = EarlyStopping(monitor='val_CRPS',
                        mode='min',
                        restore_best_weights=True,
                        verbose=0,
                        patience=10)
    
    es.set_model(model)
    metric = Metric(model, [es], [X_val, y_val_values])

    lr_i = 1e-3
    lr_f = 5e-4
    n_epochs = 50 

    decay = (1-lr_f/lr_i)/((lr_f/lr_i)* n_epochs - 1)  #Time-based decay formula
    alpha = (lr_i*(1+decay))
    
    opt = Adam(learning_rate=1e-3)
    model.compile(loss=crps,
                  optimizer=opt)
    
    model.fit(X_train,
              y_train_values, 
              epochs=n_epochs,
              batch_size=64,
              verbose=0,
              callbacks=[metric],
              validation_data=(X_val, y_val_values))

    val_crps_score = min(model.history.history['val_CRPS'])
    print("Val loss: {}".format(val_crps_score))
    
    score.append(val_crps_score)

    models.append(model)
    
print(np.mean(score))


# ===== Code cell 41 =====
print("The mean validation loss is {}".format(np.mean(score)))


# ===== Markdown cell 42 =====
# ## Conclusions


# ===== Markdown cell 43 =====
# With this elegant solution, The ZOO won the NFL Big Data Bowl 2020 (Kaggle competition). We submitted this code in kaggle and the score obtained was 0.011911 (2nd position in Leaderboard) [4]. A possible reason for the difference between our score and the winning score (0.011658) is that we do not implement TTA in our predictions. Moreover, the number of trainable parameters in our network structure differs (145,584) from the number that they report in [1] (145,329).
# 
# For further improvements, I would suggest trying to add features related to pitch control which was the VIP hint of the competition [5].


# ===== Markdown cell 44 =====
# [1] https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400
# 
# [2] https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview
# 
# [3] https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
# 
# [4] https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
# 
# [5] http://www.lukebornn.com/papers/fernandez_ssac_2018.pdf


# ===== Code cell 45 =====
# (empty cell)


# ===== Added section: bootstrap stability study scaffold =====
# The notebook above trains a convolutional model for yard distribution prediction.
# The helper functions below estimate runtime and quantify uncertainty across bootstrap refits.
import math

def estimate_bootstrap_runtime(
    n_bootstrap,
    fit_minutes_per_model,
    n_data_fractions=6,
    n_model_families=3,
    parallel_workers=1,
    overhead_fraction=0.15,
):
    """Estimate total wall-clock hours for a bootstrap + data-size stability experiment.

    Total model fits ~= n_bootstrap * n_data_fractions * n_model_families.
    Runtime includes configurable orchestration overhead.
    """
    total_fits = n_bootstrap * n_data_fractions * n_model_families
    serial_minutes = total_fits * fit_minutes_per_model
    adjusted_minutes = serial_minutes * (1 + overhead_fraction)
    wall_minutes = adjusted_minutes / max(parallel_workers, 1)
    return {
        "total_fits": int(total_fits),
        "serial_hours": serial_minutes / 60.0,
        "wall_clock_hours": wall_minutes / 60.0,
    }

def bootstrap_prediction_uncertainty(predictions_matrix):
    """Summarize prediction stability across bootstrap refits.

    Args:
        predictions_matrix: numpy array with shape [n_bootstrap, n_examples].
    Returns:
        Per-example variance and central 90% interval width across bootstrap models.
    """
    import numpy as np
    var_by_example = np.var(predictions_matrix, axis=0)
    q95 = np.quantile(predictions_matrix, 0.95, axis=0)
    q05 = np.quantile(predictions_matrix, 0.05, axis=0)
    width90_by_example = q95 - q05
    return {
        "variance_mean": float(np.mean(var_by_example)),
        "variance_median": float(np.median(var_by_example)),
        "width90_mean": float(np.mean(width90_by_example)),
        "width90_median": float(np.median(width90_by_example)),
    }

# Example experiment sizing:
# runtime = estimate_bootstrap_runtime(
#     n_bootstrap=100, fit_minutes_per_model=20, n_data_fractions=6, n_model_families=3,
#     parallel_workers=4, overhead_fraction=0.20
# print(runtime)