# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/mspoorendonk/drivendata/blob/marc/drivendata_waterpump.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="pC9Hngyu5E6R"
# # Analysis of condition of water points in Tanzania
#
# Problem statement:
# predict the operating condition of a waterpoint for each record in the dataset: functioning, functioning but needs repair, not functioning
#
#
# Approach
# 1. Download datasets
# 1. Explore data and understand which features are relevant for the prediction. 
# 1. Clean data [Bart]
# 1. Engineer some derived features
# 1. decide on a method for predicting (trees or neuralnets or knn or ...)
# 1. perform a train / test / validate split on the data
# 1. Train model on training values and labels
# 1. Predict training labels that correspond to training values
# 1. Report the accuracy
# 1. Tune hyperparameters with gridsearch
# 1. Predict the test labels
# 1. Submit CSV [Marc]
#
#
# TODO:
# here: check xgboost, pandas, bokeh (interactief)
# somewhere else: how to deploy a model in production. What software and frameworks etc.
#

# %% [markdown] id="hwqftAGR-k2J"
# # Dependencies

# %% id="PuCGSSOKu8-h" outputId="4e1ba0b1-c2e7-415d-aa7f-0f9f9b61db51" colab={"base_uri": "https://localhost:8080/", "height": 728}
# installations

# !pip install gmaps

# %% id="CJlE0N65YUFd" outputId="d185adbf-f040-41dc-fac3-18b6b92f5566" colab={"base_uri": "https://localhost:8080/", "height": 52}
# imports

from datetime import datetime
import pandas as pd
import random
import numpy as np
import gmaps
import IPython
from sklearn import tree # to create a decision tree

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics # to compute accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import preprocessing # for normalizing data for knn
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler
import pydotplus # To create our Decision Tree Graph
from IPython.display import Image  # To Display a image of our graph
from IPython.display import display

from ipywidgets.embed import embed_minimal_html

# Seaborn visualization library
import seaborn as sns # for pairplots
import matplotlib.pyplot as plt

# %% [markdown] id="mOSkSXlq-gzD"
# # Download datasets

# %% id="tGa2jMrp23Sm" outputId="7c82da7a-96fa-4ae3-9b26-f360fb799bd1" colab={"base_uri": "https://localhost:8080/", "height": 357}
# download datasets from driven-data.org. Urls copied from data download section on website.
# They expire after 2 days or so. Then you need to copy/paste them again.

# testvalues
# !wget "https://drivendata-prod.s3.amazonaws.com/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCY3EFSLNZR%2F20200927%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200927T185304Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f2b7c554cb780a1facf849dc85cd18a0ce5110100690a748eaa1df42f43a12da" -O test_values.csv
# training labels
# !wget "https://drivendata-prod.s3.amazonaws.com/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCY3EFSLNZR%2F20200927%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200927T185304Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f91daa03811de5cb244f5f2d8446fb46a99eb37bedf7bd0c609d8b076bebfbe2" -O training_labels.csv
# training values
# !wget "https://drivendata-prod.s3.amazonaws.com/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCY3EFSLNZR%2F20200927%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200927T185304Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=b38e27dab8fac51df99d1ec837ffd2f4a3c3e1ffd48494951a846a144f88434f" -O training_values.csv

# %% id="s5JdImi9qyql"
# Boundary coordinates of Tanzania
# Source: https://en.wikipedia.org/wiki/List_of_countries_by_northernmost_point (and similar)
tanzania_lat = [-11.750-0.1, -0.983+0.1]
tanzania_lon = [29.167-0.1, 40.250+0.1]

# %% pycharm={"name": "#%%\n"}
# Data location
data_path = ''


# %% id="TlOL83NOOp9n" outputId="f1473e4e-a572-42ed-9413-b57797dcdc6d" colab={"base_uri": "https://localhost:8080/", "height": 369}
# Load training values
na_values = {
    'longitude': 0.0,
    'latitude':-2.e-8,
    'gps_height': 0,
#     'wpt_name': 'none',
#     'construction_year': 0,
#     'population': 0,
#     'district_code': 0,
}
# na_values = {}
training_values = pd.read_csv(data_path + 'training_values.csv',
                              parse_dates=['date_recorded'],
                              index_col='id',
                              na_values=na_values)
# Drop column(s) without information
training_values.drop(columns=['num_private'], inplace=True)
print('Shape: ', training_values.shape)
# Show example
display(training_values.iloc[:5, 0:20])
display(training_values.iloc[:5, 20:])

# %% pycharm={"name": "#%%\n"}
# Load training labels
training_labels = pd.read_csv(data_path + 'training_labels.csv',
                             index_col='id')
training_labels.head()

# %% pycharm={"name": "#%%\n"}
# Merge training values and -labels
training_all = pd.merge(training_values, training_labels, on='id')
training_all.head()

# %% pycharm={"name": "#%%\n"}
# Load submission values
test_values = pd.read_csv(data_path + 'test_values.csv',
                          parse_dates=['date_recorded'],
                          index_col='id',
                          na_values=na_values)

# %% id="e1LWiBzgPUyv"
# column_names = ''
# for n in training_values.columns:
#     column_names = column_names + "'" + n + "', "
# print(column_names) # print a string from which we can copy/paste the following lists

columns_time = ['date_recorded']
columns_numerical = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
             'construction_year', 'region_code', 'district_code']
columns_categorical = ['funder', 'installer', 'wpt_name', 'basin', 'subvillage',
               'region', 'lga', 'ward', 'public_meeting', 'recorded_by',
               'scheme_management', 'scheme_name', 'permit', 'extraction_type', 
               'extraction_type_group', 'extraction_type_class', 'management', 
               'management_group', 'payment', 'payment_type', 'water_quality', 
               'quality_group', 'quantity', 'quantity_group', 'source', 'source_type', 
               'source_class', 'waterpoint_type', 'waterpoint_type_group']
columns_location = ['latitude', 'longitude', 'gps_height', 'wpt_name', 'basin', 'subvillage',
                    'region', 'region_code', 'district_code', 'lga', 'ward']

print('Time: ', len(columns_time))
print('Numerical: ', len(columns_numerical))
print('Categorical: ', len(columns_categorical))

# %% [markdown]
# # Exploratory Data Analysis
#

# %% pycharm={"name": "#%%\n"}
# Show main training data characteristics
training_values.info()
training_values.describe()

# %% pycharm={"name": "#%%\n"}
# Investigate duplicate rows
headers = list(training_values)
duplicate_full = training_values.duplicated(subset=headers, keep=False)
duplicate_full_all = training_all.duplicated(subset=(headers + ['status_group']), keep=False)
print('Number of fully duplicate rows: ', duplicate_full.sum())
print('Number of fully duplicate rows (incl label): ', duplicate_full_all.sum())
print('Examples of duplicate rows:')
display(training_values[duplicate_full].sort_values(by=headers).head(10))

# Find the rows that are duplicate apart from the label
id_diff = set(training_values[duplicate_full].index).difference(training_all[duplicate_full_all].index)
print('Duplicate apart from label:')
display(training_all.loc[id_diff].head(10))

# Mark duplicates for removal
flag_droprows = training_values.sort_index().duplicated(subset=headers, keep='first')
# Add both rows with different label
flag_droprows[id_diff] = True
print('Marked for removal: {}'.format(flag_droprows.sum()))

# %% pycharm={"name": "#%%\n"}
# Investigate Total static head (amount_tsh)
# Note that the units are unknown
fig, ax = plt.subplots()
training_values['amount_tsh'].plot.hist(ax=ax, log=True, bins=20)
n_zero_tsh= (training_values['amount_tsh']==0).sum()
n_total = training_values.shape[0]
print('Rows where amount_tsh == 0.0: {}, ({:3d}%)'\
      .format(n_zero_tsh,int(round(100*n_zero_tsh/n_total, 2))))
# Conclusion: amount_tsh looks okay, some zeros might actually be missing
# values, but cannot be distinguished from measured zeros.

# %% pycharm={"name": "#%%\n"}
# Investigate date_recorded
fig, ax = plt.subplots()
training_values['date_recorded'].dt.year.plot.hist(ax=ax)
display(training_values['date_recorded'].dt.year.value_counts(sort=False))
ax.set_title('Histogram of date_recorded')
year2011 = (training_values['date_recorded'].dt.year == 2011)
year2012 = (training_values['date_recorded'].dt.year == 2012)
year2013 = (training_values['date_recorded'].dt.year == 2013)
fig, ax2 = plt.subplots()
training_values[year2011]['date_recorded'].dt.month.plot.hist(ax=ax2, bins=12,
    range=(1, 12), log=True)
ax2.set_title('2011, records per month')
fig, ax3 = plt.subplots()
training_values[year2012]['date_recorded'].dt.month.plot.hist(ax=ax3, bins=12,
    range=(1, 12), log=True)
ax3.set_title('2012, records per month')
fig, ax4 = plt.subplots()
training_values[year2013]['date_recorded'].dt.month.plot.hist(ax=ax4, bins=12,
    range=(1, 12), log=True)
ax4.set_title('2013, records per month')
# Conclusion: dates look OK.

# %% pycharm={"name": "#%%\n"}
# Investigate GPS height
# Note that GPS height is inaccurate, deviations of 120 m are not uncommon.
height_neg = training_values['gps_height'] < 0
height_pos = training_values['gps_height'] > 0
height_zero = training_values['gps_height'] == 0
fig, ax = plt.subplots()
training_values['gps_height'].plot.hist(ax=ax)
ax.set_title('Histogram of GPS Height')
fig, ax = plt.subplots()
training_values[height_neg]['gps_height'].plot.hist(ax=ax)
ax.set_title('Histogram of GPS Height (strictly negative)')
print('Rows with zero height: {}'.format(height_zero.sum()))
print('Rows with negative height: {}'.format(height_neg.sum()))
fig, ax = plt.subplots()
training_values['gps_height'].plot.hist(ax=ax, range=(-20,20))
print(training_values[~height_zero]['gps_height'].median())
# Conclusion: GPS height looks OK. Negative values can be explained by
# inaccuracy of measurement. Zeros are most likely missing values,
# since they occur much more frequently than other values.

# %% pycharm={"name": "#%%\n"}
# Investigate region
# Note: Tanzania has 31 regions, 169 districts (2012)
# https://en.wikipedia.org/wiki/Districts_of_Tanzania
region_counts = training_values[['region_code','region']]\
    .sort_values(by='region').value_counts(sort=False)
display(region_counts)
display(training_values.query('region_code==5 & region=="Tanga"')\
    .loc[:, columns_location].head(10))
# Conclusion: some region codes (5, 11, 14, 17, 18) seem to refer to multiple
# regions. However, if the region with the highest count is correct, this
# affects only 123 rows. Some regions (Arusha, Lindi, Mtwara, Mwanza, Pwani,
# Shinyanga, Tanga) have multiple region codes associated to them.
# Assuming that the most common mapping is correct, this affects around 2500 rows.
# Solutions:
# - Use only latitude and longitude as location data. However, the region,
#   district, lga, ward might contain information about governance.
# - Remove dubious location data and/or mark as missing.
# - Use an external source (e.g. GeoNames.org) to verify which values are
#   most likely incorrect and replace those.

# %% pycharm={"name": "#%%\n"}
# Investigate district
# Note: according to wikipedia, regions have up to 10 districts
display(training_values.sort_values(by='district_code')\
    .value_counts(subset='district_code',sort=False))
display(training_values.sort_values(by='district_code')\
    .value_counts(subset=['region', 'district_code'],sort=False))
# Investigate example of district_code > 10
display(training_values.query('district_code==80')\
    .loc[:,['region','region_code','district_code','subvillage']].head(12))
# Conclusion: all district codes larger than 10 are most probably duplicates of
# other codes. This affects some 4200 rows. The tuple (region_code,
# district_code) should uniquely identify a region, so we can derive a new
# feature based on these codes.

# %% pycharm={"name": "#%%\n"}
# Investigate location (latitude, longitude)

# Check if latitude, longitude is actually inside Tanzania (or NaN)
lon_isna = training_values['longitude'].isna()
lat_isna = training_values['latitude'].isna()
lon_in_range = (tanzania_lon[0] <= training_values['longitude']) & \
               (training_values['longitude'] <= tanzania_lon[1])
lat_in_range = (tanzania_lat[0] <= training_values['latitude']) & \
               (training_values['latitude'] <= tanzania_lat[1])
pos_isna = lon_isna | lat_isna
pos_in_range = lon_in_range & lat_in_range
print('Number of missing (n/a) coordinates: ', pos_isna.sum())
print('Number of invalid coordinates: ', (~pos_in_range & ~pos_isna).sum())

# Investigate duplicate locations
duplicate_location = training_values.duplicated(
    subset=['longitude', 'latitude'], keep=False)
print('Number of rows with duplicate locations: ',
    (duplicate_location & ~pos_isna & ~duplicate_full).sum())
training_values[duplicate_location & ~pos_isna & ~duplicate_full]\
    .sort_values(['latitude', 'longitude']).head(10)

# %% id="XXko4e2zOyim"
training_values.describe()

# %% id="u637XYufRh8f"
training_labels

# %% id="wyL52BBIFcsH"


# Create the default pairplot
# sns.pairplot(training_all[columns_numerical + ['status_group']], hue = 'status_group')

# %% [markdown] id="kPHHIaJI-tVR"
# # Engineer features

# %% id="XscRcgEZRYuQ"
# engineer some features

# maybe days since reporting a functional pump?
# lifetime: date_recorded - construction_year

# %% [markdown] id="vI-9_Bsg-zaK"
# # Clean data

# %%
for column in ['latitude', 'longitude', 'gps_height']:
    for data in [training_values, test_values]:
        # Impute, stepwise from specific to general (ward > lga > region > column)
        data[column].fillna(data.groupby(['region', 'lga', 'ward'])[column].transform('mean'), inplace=True)
        data[column].fillna(data.groupby(['region', 'lga'])[column].transform('mean'), inplace=True)
        data[column].fillna(data.groupby(['region'])[column].transform('mean'), inplace=True)
        data[column].fillna(data[column].mean(), inplace=True)
display(training_values[['latitude', 'longitude', 'gps_height']].isna().sum())
display(test_values[['latitude', 'longitude', 'gps_height']].isna().sum())

# %% id="uLzANEKyRuS9"
# plot n pumps on a map. Everything above 200 gets slow

n = 200

gmaps.configure(api_key="AIzaSyCDAaxun4CXAyEmLzzJbYkqXii-sbVhVNc")  # This is my personal API key, please don't abuse.



colors = []
labels = []


sampled_pumps = training_values.sample(n)

for i in range(len(sampled_pumps)):
  id = sampled_pumps.iloc[i]['id']
  #print(id)
  state = training_labels[training_labels['id']==id]['status_group'].iloc[0]
  if state=='functional':
    colors.append('green')
  elif state=='non functional':
    colors.append('red') 
  else:
    colors.append('yellow') # needs repair

  labels.append('source %s' % sampled_pumps[sampled_pumps['id']==id].iloc[0]['source'])


pump_locations = sampled_pumps[['latitude' , 'longitude']]
info_box_template = """
<dl>

<td>Name</td><dd>{scheme_name}</dd>
</dl>
"""

pump_info = training_values['scheme_name'][:2]

#marker_layer = gmaps.marker_layer(pump_locations, hover_text=pump_info, info_box_content=pump_info)
marker_layer = gmaps.symbol_layer(pump_locations, fill_color=colors, stroke_color=colors, scale=3, hover_text=labels)
figure_layout = {
    'width': '1400px',
    'height': '1200px',
    'border': '1px solid black',
    'padding': '1px'
}

fig = gmaps.figure(layout=figure_layout)
fig.add_layer(marker_layer)
#fig
embed_minimal_html('export.html', views=[fig])
IPython.display.HTML(filename='export.html')

# %% id="2oP1BmwhaoD9"
training_values[['longitude', 'latitude']].head()

# %% [markdown] id="eFoRLkhd-5Ca"
# # Prepare for training

# %% id="rd8zedFQX7JF"

# set n to low number for faster runs and to len(training_values) for max accuracy
# n = 5000
n = len(training_values)
# select the describing variables
# columns_select = [
#                   'id',
#                   'date_recorded',
#                   'amount_tsh',
#                   'gps_height',
#                   'longitude',
#                   'latitude',
#                   'region_code',
#                   'district_code',
#                   'population',
#                   'construction_year',
#                   'source',
#                   'quality_group',
#                   'quantity_group',
#                   'extraction_type_group',
#                   ]
columns_select = [
                  'date_recorded',
                  'amount_tsh',
                  'gps_height',
                  'longitude',
                  'latitude',
                  'region_code',
                  'district_code',
                  'population',
                  'construction_year',
                  'source',
                  'source_class',
                  'management_group',
                  'payment_type',
                  'extraction_type_group',
                  'waterpoint_type_group',
                  'quality_group',
                  'quantity_group',
                  'extraction_type_group',
                 ]
X = pd.get_dummies(training_values[columns_select][:n])
#X = pd.get_dummies(training_values[:n])
# X=X.drop(X[X['construction_year']< 1900].index) # drop all lines with missing construction year (but thenalso drop the y!!)
X['lifetime']=pd.DatetimeIndex(X['date_recorded']).year-X['construction_year']  # engineer a feature but don't do it for rows where construction_year is empty
X.loc[X['lifetime']> 1000, 'lifetime']=-1
X['date_recorded']=X['date_recorded'].apply(datetime.toordinal) # otherwise dates get ignored in the correlation and the tree

enc = preprocessing.LabelEncoder()
Y = enc.fit_transform(np.ravel(training_labels[['status_group']][:n]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

scaler = MinMaxScaler()
scaler.fit(X)
X_train_normalized = scaler.transform(X_train)
X_test_normalized  = scaler.transform(X_test)

# Load and transform verification/submission data
X_submission = pd.get_dummies(test_values[columns_select][:n])
X_submission['lifetime']=pd.DatetimeIndex(X_submission['date_recorded']).year-X_submission['construction_year']
X_submission.loc[X_submission['lifetime']> 1000, 'lifetime']=-1
X_submission['date_recorded']=X_submission['date_recorded'].apply(datetime.toordinal) # otherwise dates get ignored in the correlation and the tree



# %% id="Ux1_vI5d2jbe"



# %% id="wu8gomyui65T"
Y_train

# %% id="qPhexPmsgZ0-"
np.array(X_train['lifetime'][:100])


# %% id="KqlNFS1OUrAN"
# figure out which variables correlate with Y


sns.set(rc={'figure.facecolor':'#a0a0a0'})

XY=pd.concat([X, Y], axis=1) # get them side by side

corrMatrix = XY.corr()
plt.figure(figsize=(60,25))
# for tips on formatting the heatmap:
# https://heartbeat.fritz.ai/seaborn-heatmaps-13-ways-to-customize-correlation-matrix-visualizations-f1c49c816f07
sns.heatmap(corrMatrix, annot=True,  fmt='.2f', vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.show()


# %% [markdown] id="FnfLpQ-IRJuI"
# #Forecast

# %% id="21pkGct572-T"
def calc_accuracy(y_pred, Y_test):
  correct = 0
  for i in range(len(y_pred)):
    y_vals = Y_test.iloc[i].values
    y_pred_vals = y_pred[i]
    #print(y_vals, y_pred_vals)
    if (y_vals == y_pred_vals).all():
      #print("correct")
      correct += 1
    #else:
      #print('incorrect')
    #if correct>10: break
  return correct/len(y_pred), correct

results = {}

# %% [markdown] id="SyNH4rTsRlO0"
# ##Decision tree

# %% id="OZDyxJP1lXAp"
print("Train on %d samples. Test on %d samples." % (len(X_train), len(X_test)))

results['tree'] = 0
for d in [1, 5, 10, 15, 20, 25, 5]: # end with 5 so it can be plotted in next cell
  model = tree.DecisionTreeClassifier(criterion='gini',max_depth=d)
  model = model.fit(X_train, Y_train)

  #Predict the response for test dataset
  y_pred = model.predict(X_test)
 
  accuracy, correct=calc_accuracy(y_pred, Y_test)
  print("Max depth: %d   Accuracy on test set: %.2f   #correct: %d" % (d, accuracy, correct))
  if accuracy > results['tree']: results['tree']=accuracy

# %% id="nQp0frzjlYo_"
# Export/Print a decision tree in DOT format. Only do this when max_depth is small (<=6) otherwise it gets too slow.
#print(tree.export_graphviz(clf, None))

if d < 6:
  print('extracting dot')
  #Create Dot Data
  dot_data = tree.export_graphviz(model, out_file=None, feature_names=list(X_train.columns.values), 
                                  class_names=['func', 'repair', 'nonfunc'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
  #print(dot_data)
  print('Create graph image from DOT data')
  graph = pydotplus.graph_from_dot_data(dot_data)

  print('Show graph')
  Image(graph.create_png())
else:
  print('graph to deep to fit in image')

# %% [markdown] id="TitZ_1pRbHIu"
# ##Random forest

# %% id="cwKWxHMmasjX"
print("Train on %d samples, %d variables. Test on %d samples." % (X_train.shape[0], X_train.shape[1], len(X_test)))

d=25
model = RandomForestClassifier(n_jobs=-1, random_state=27, verbose=0, max_depth=d, criterion='gini')
model = model.fit(X_train, np.ravel(Y_train))

#Predict the response for test dataset
y_pred = model.predict(X_test)

# accuracy, correct=calc_accuracy(y_pred, Y_test)
# print("Max depth: %d   Accuracy on test set: %.2f   #correct: %d" % (d, accuracy, correct))
accuracy = accuracy_score(Y_test, y_pred)
print("Max depth: %d   Accuracy on test set: %.3f" % (d, accuracy))
# results['forest']=accuracy


# %% id="9FnQqh1FABWE" outputId="1f001ef9-2481-489d-d94f-e8071f23bebe" colab={"base_uri": "https://localhost:8080/", "height": 595}
# feature importances
#inspiration: https://github.com/ernestng11/touchpoint-prediction/blob/master/model-building.ipynb

print(len(model.feature_importances_))
combined = zip(model.feature_importances_, X_train.columns)
combined = sorted(combined, reverse=True)[:50]
#print(combined)
#for i in len(combined):
#  print('%s\t%.3f' % (combined[i][1], combined[i][0]))

importance, features = list(zip(*combined))

f, ax = plt.subplots(figsize=(35,5))
plot = sns.barplot(x=np.array(features), y=np.array(importance))
ax.set_title('Feature Importance')
plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
plt.show()

# %%
# Permutation importance
# To prevent biased towards high-cardinality features and 
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, Y_test,
                                n_repeats=10,
                                scoring='accuracy',
                                random_state=0)

for i in result.importances_mean.argsort()[::-1]:
    if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
        print(f"{X_test.columns[i]:<8}"
              f"{result.importances_mean[i]:.3f}"
              f" +/- {result.importances_std[i]:.3f}")

sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx[-20:]].T,
           vert=False, labels=X_test.columns[sorted_idx[-20:]])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()

# %% [markdown] id="aZ0aE1t8RRTO"
# ##KNN

# %% id="hBkoB37GCayu" outputId="337b4e6e-c637-42c3-b1cb-4c1c962e4c04" colab={"base_uri": "https://localhost:8080/", "height": 177}
print("Train on %d samples. Test on %d samples." % (len(X_train), len(X_test)))


results['knn']=-1
for d in [1, 2, 3, 5, 10, 15, 20, 30]:
  model = KNeighborsClassifier(n_neighbors=d)
  model = model.fit(X_train_normalized, Y_train)

  #Predict the response for test dataset
  y_pred = model.predict(X_test_normalized)

  accuracy, correct=calc_accuracy(y_pred, Y_test)
  print("n_neighbors: %d   Accuracy on test set: %.2f   #correct: %d" % (d, accuracy, correct))
  if accuracy > results['knn']: results['knn']=accuracy

# %% id="L-cFwS5CHzMB" outputId="7c71f624-fdb3-4ebf-fc2e-975889b2a920" colab={"base_uri": "https://localhost:8080/", "height": 411}
pd.DataFrame( Y_train)

# %% [markdown] id="1gM_orokRXhO"
# ##Neuralnet

# %% id="KPk1W5P-E4iv" outputId="a505866d-3612-4af7-b94b-fb537c7bca23" colab={"base_uri": "https://localhost:8080/", "height": 1000}
print("Train on %d samples. Test on %d samples." % (len(X_train), len(X_test)))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
#model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(20,  activation="relu", input_shape = (X_test_normalized.shape[1],)))
model.add(layers.Dense(10,  activation="relu"))
model.add(layers.Dense(5,  activation="relu"))
model.add(layers.Dense(3,   activation='sigmoid'))
model.compile('adam', "binary_crossentropy", metrics=["accuracy"])
model.fit(x=X_train_normalized, y=Y_train, epochs=35)
model.summary()

y_pred = model.predict(X_test_normalized)
print(len(y_pred))
y_pred = (y_pred > 0.5).astype("int32")

accuracy, correct=calc_accuracy(y_pred, Y_test)
print("Accuracy on test set: %.2f   #correct: %d" % (accuracy, correct))
results['neural net']=accuracy

# %% [markdown] id="GSrcI0Gae9cE"
# ##XGBoost

# %% id="YK24eHPVfFQw" outputId="22dba33c-d6ce-49c8-a341-556ae4bbb141" colab={"base_uri": "https://localhost:8080/", "height": 35}
# inspiration: https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn

from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer


#for d in range(1,35):
results['xgboost']=-1
#for d in [2, 15, 30, 50]:
for d in [30]:
  model = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=d, objective="multi:softprob", num_class=3))
  model = model.fit(X_train_normalized, Y_train)

  #Predict the response for test dataset
  y_pred = model.predict(X_test_normalized)

  accuracy, correct=calc_accuracy(y_pred, Y_test)
  print("XGBoost: max_depth: %d   Accuracy on test set: %.2f   #correct: %d" % (d, accuracy, correct))
  if accuracy>results['xgboost']: results['xgboost']=accuracy

# %% id="wtt6Yw9AZMoc"
#print(confusion_matrix(Y_test, y_pred))

# %% [markdown] id="OKXg_c92EjZZ"
# #Evaluation
# - randomforest: .72 
# - tree: .70
# - xgboost: .70
# - nn: .65
# - knn: .48

# %% id="eiN5HwCD9cEC"
for k in results.keys():
  print('%s: %.2f' % (k.capitalize(), results[k]))



# %% id="q6LPzRASj4M5"
import requests
gcloud_token = !gcloud auth print-access-token
gcloud_tokeninfo = requests.get('https://www.googleapis.com/oauth2/v3/tokeninfo?access_token=' + gcloud_token[0]).json()
gcloud_tokeninfo


# %% [markdown] id="wHrNq7Z1PuQt"
# #Submit result

# %% id="G0ag_iZe-aE3"
print('train model')

model = RandomForestClassifier(n_jobs=None,random_state=27,verbose=0, max_depth=25, criterion='gini')
# re-train on the full training set
model = model.fit(X, np.ravel(Y))

print('predict')

#Predict the response for test dataset
y_pred = model.predict(X_submission)

print('create submission')
# create a dataframe for submission
submission = pd.DataFrame(columns=['id', 'status_group'])
status = enc.inverse_transform(y_pred)
submission['status_group'] = status
submission['id'] = test_values.index
display(submission.head())
submission['status_group'].value_counts().plot(kind='bar')

# save as csv
submission.to_csv('submission.csv', index=False)

# %% id="-Qv6DQ-SS5Gi"
test_values

# %% id="5Vdfr1-xWIJD"

# %% [markdown] id="Ww0xXVHe201e"
# # Graveyard
# Snippets that are incomplete but interesting nonetheless

# %% id="KwCo_6lJCtV1"
# inspired by: https://medium.com/@gabrielziegler3/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d

from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

model = XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=1000, 
                        num_classes=3)
model = model.fit(X_train_normalized, Y_train)




#Predict the response for test dataset
y_pred = model.predict(X_test_normalized)

accuracy=calc_accuracy(y_pred, Y_test)
print("n_neighbors: %d   Accuracy on test set: %.2f   #correct: %d" % (d, accuracy, correct))
accuracy_xgboost=accuracy
