
# Disclaimer : This File contains a plot of World Map which may take 3-4 hours to run the complete file. Please comment out the plot (line: 493) if not helpful


import os
import sys
import pandas as pd 
import numpy as np
import seaborn as sns
import geopandas as gpd
import geopandas as gpd
import contextily as ctx
from sklearn import metrics
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.pipeline import Pipeline
sys.path.append(os.path.abspath(".."))
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from shapely.geometry import Polygon, LineString, Point
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression



# Utility Functions Used in the process


def missing_report(df, top_n=None, percent=True, sort=True, threshold=None):
    """
    Generates a bar plot showing missing values per column (as count or percentage)
    and prints a list of column names whose missing values exceed a given threshold or fall
    within a provided range.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - top_n (int, optional): Show only the top N columns with the most missing data.
    - percent (bool): If True, calculates missing values as a percentage; otherwise, as counts.
    - sort (bool): Sort the missing values in descending order.
    - threshold (float or tuple/list of two floats, optional): 
        If a single float, prints a list of columns with missing values above this threshold.
        If a tuple/list of two floats, prints a list of columns with missing values between the lower and upper bounds.
    
    Returns:
    - list: A list of column names that meet the threshold criteria.
    """
    if percent:
        missing = (df.isnull().sum() / len(df)) * 100
        ylabel = 'Missing (%)'
    else:
        missing = df.isnull().sum()
        ylabel = 'Missing Count'
    
    if sort:
        missing = missing.sort_values(ascending=False)
    if top_n:
        missing = missing.head(top_n)
    
    above_threshold = None
    
    if threshold is not None:
        if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
            lower, upper = threshold
            above_threshold = missing[(missing >= lower) & (missing <= upper)].index.tolist()
            print(f"Columns with missing values between {lower} and {upper} {ylabel}:")
        else:
            above_threshold = missing[missing > threshold].index.tolist()
            print(f"Columns with missing values above {threshold} {ylabel}:")
        print(above_threshold)
    
    
    plt.figure(figsize=(12, 8))
    missing.plot(kind='bar')
    plt.ylabel(ylabel)
    plt.title('Missing Data Report')
    plt.xlabel('Columns')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return above_threshold




def get_binary_and_continuous_columns(df, binary_threshold=2, unique_cutoff=3):
    """
    Identifies binary and continuous numeric columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - binary_threshold (int): Number of unique values to consider as binary (default=2)
    - unique_cutoff (int): Minimum number of unique values to consider a column continuous (default=10)

    Returns:
    - dict: {
        'binary': [list of column names],
        'continuous': [list of column names]
      }
    """
    binary_cols = []
    continuous_cols = []

    for col in df.select_dtypes(include=['number']).columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == binary_threshold:
            binary_cols.append(col)
        elif len(unique_vals) >= unique_cutoff:
            continuous_cols.append(col)

    return {
        'binary': binary_cols,
        'continuous': continuous_cols
    }



def plot_boxplots(df, columns, figsize=(12, 6), rotation=45):
    """
    Plots boxplots for a list of numeric columns from a DataFrame.

    Parameters:
    - df (pd.DataFrame): Your data
    - columns (list): List of column names to plot
    - figsize (tuple): Size of the plot canvas
    - rotation (int): Angle of x-axis labels
    """
    num_cols = len(columns)
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(figsize[0] * num_cols, figsize[1]))

    if num_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        sns.boxplot(data=df, y=col, ax=ax)
        ax.set_title(col)
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=rotation)

    plt.tight_layout()
    plt.show()




def get_statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing statistical analysis of the input DataFrame.
    
    The summary includes:
      - Datatype of the column
      - Mean (if numeric)
      - Median (if numeric)
      - Count of unique values
      - Count of null/NA values
      - Percentage of null/NA values
      - Standard deviation (if numeric)
      - Variance (if numeric)
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: A summary DataFrame with the statistics for each column.
    """
    
    summary_list = []
    
    
    for col in df.columns:
        data = df[col]
        col_dtype = data.dtype
        unique_count = data.nunique(dropna=True)
        null_count = data.isna().sum()
        null_percentage = (null_count / len(data)) * 100
        
        
        if pd.api.types.is_numeric_dtype(data):
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            var_val = data.var()
        else:
            mean_val = np.nan
            median_val = np.nan
            std_val = np.nan
            var_val = np.nan
        
        summary_list.append({
            "Column": col,
            "Datatype": col_dtype,
            "Mean": mean_val,
            "Median": median_val,
            "Unique Count": unique_count,
            "Null Count": null_count,
            "Null Percentage": null_percentage,
            "Std": std_val,
            "Variance": var_val
        })
    
    
    summary_df = pd.DataFrame(summary_list)
    
    summary_df.set_index("Column", inplace=True)
    
    return summary_df


def apply_knn_imputer(df, columns=None, n_neighbors=5):

    """
    Applies KNN imputation to specified columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame
    - columns (list, optional): Subset of columns to impute. If None, all numeric columns are used.
    - n_neighbors (int): Number of neighbors to use for imputation

    Returns:
    - pd.DataFrame: New DataFrame with imputed values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include='number').columns.tolist()
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_copy[columns] = imputer.fit_transform(df_copy[columns])
    
    return df_copy



def smart_impute(df, cols, group_col=None, skew_threshold=1):
    """
    Imputes missing values in specified numeric columns using group-specific statistics if group_col is provided,
    otherwise performs simple imputation based on overall column skew.
    
    For each column:
      - Only numeric columns are imputed.
      - If group_col is provided, groups the DataFrame by `group_col` and fills missing values with:
          - The group's median if the column's skew > skew_threshold,
          - Otherwise, the group's mean.
      - If group_col is None, fills missing values with:
          - The column's median if its skew > skew_threshold,
          - Otherwise, the column's mean.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - cols (list): List of column names to impute.
    - group_col (str, optional): Column name to group by. If not provided, performs global imputation.
    - skew_threshold (float): Threshold for skew to decide median vs. mean imputation (default 1).
    
    Returns:
    - pd.DataFrame: A new DataFrame with imputed values for numeric columns.
    """
    df_copy = df.copy()
    
    for col in cols:
        
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
        
        if group_col is not None:
            df_copy[col] = df_copy.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median() if x.skew() > skew_threshold else x.mean())
            )
        else:
            if df_copy[col].skew() > skew_threshold:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    return df_copy



def cat_data(df):
    '''Generate number of unique values and catplot of columns in df classified as category'''
    print(df.select_dtypes(include=['category']).nunique())
    for col in df.select_dtypes(include='category').columns:
        fig = sns.catplot(x=col, kind='count', data=df)
        fig.set_ylabels('Count')
        fig.set_xlabels('')
        fig.set(title=col)
        fig.set_xticklabels()
        plt.show()

def num_data(df):
    '''Generate boxplot in df from numerical column data'''
    for col in df.select_dtypes(include=['int64','float']).columns:
        fig = sns.boxplot(x=col, orient='v', data=df)
        plt.show()



def lmplot(data, x, y, xlabel, ylabel, title, height=12, aspect=1, theme='poster', target='LILATracts_halfAnd10',\
          style='darkgrid'):
    '''Creates lmplot to comepare two variables vs the target 
    Enter dataframe, x, y, xlabel, ylabel, title.
    Height and aspect have default values
    Seaborn theme default poster, theme to darkgrid
    Target default to LILATracts_halfAnd10'''
    sns.set_style(style)
    sns.set_theme(theme)
    sns.lmplot(x=x, 
               y=y,  
               data=data,
              height=12,
              aspect=1,
               legend_out=False,
              hue=target)\
        .set(ylabel=ylabel, 
             xlabel=xlabel, 
             title=title)\
        ._legend.set_title('Target')
    plt.show();

def test():
    print('Hello World from utility.py')
    return 0


def confusion_matrix(estimator, X, y, title, display_labels=['Not Food Desert', 'Food Desert'], normalize=None):
    '''Plots a confusion matrix for a given classifier.
    
    Parameters:
    - estimator: Trained model
    - X: Features
    - y: True labels
    - title: Title for the plot
    - display_labels: Labels for the classes (default: binary)
    - normalize: 'all', 'true', 'pred', or None
    '''
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(12, 10))

    ConfusionMatrixDisplay.from_estimator(
        estimator,
        X,
        y,
        display_labels=display_labels,
        cmap=plt.cm.Blues,
        ax=ax,
        normalize=normalize
    )

    ax.set_ylabel('True Label', fontsize=18)
    ax.set_xlabel('Predicted Label', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()



# Data Preprocessing


FoodAccessAtlas = pd.read_csv("../DATA/FoodAccessResearchAtlasData2019.csv")
FoodAccessAtlas.head()

len(FoodAccessAtlas.columns.tolist())

FoodEnvironmentAtlas = pd.ExcelFile("../DATA/FoodEnvironmentAtlas.xls")
FoodEnvironmentAtlas.sheet_names

type(FoodEnvironmentAtlas)


for sheet in FoodEnvironmentAtlas.sheet_names:
    if sheet == " Variable List":
        continue
    elif sheet == "Read_Me":
        continue    
    else:
        df  = FoodEnvironmentAtlas.parse(sheet)
        df.to_csv("../DATA/FoodEnvironmentAtlas" + sheet + ".csv", index=False)



FoodAccessAtlas.info(verbose=True, show_counts=True)
missing_report_df = missing_report(FoodAccessAtlas, top_n=None)

Columns_to_drop = missing_report_df[missing_report_df["Missing (%)"] >= 50].index.tolist()
Columns_to_drop.__len__()

FoodAccessAtlascleaned = FoodAccessAtlas.drop(columns=Columns_to_drop, inplace=False)
FoodAccessAtlascleaned.shape
FoodAccessAtlascleaned.info(verbose=True, show_counts=True)

binary_cols = get_binary_and_continuous_columns(FoodAccessAtlascleaned)["binary"]
continious_col = get_binary_and_continuous_columns(FoodAccessAtlascleaned)["continuous"]

missing_report_df = missing_report(FoodAccessAtlascleaned, top_n=None, percent=True)
missing_report_df.shape

Cols_to_Impute = missing_report(FoodAccessAtlascleaned, threshold=(0, 30))
FoodAccessAtlascleaned = smart_impute(FoodAccessAtlascleaned, cols = Cols_to_Impute, group_col="County")

knn_imputing_cols = missing_report(FoodAccessAtlascleaned, threshold=30)
FoodAccesResearch_df = apply_knn_imputer(FoodAccessAtlascleaned, knn_imputing_cols)

# Creating a target variable 

FoodAccessAtlascleaned['is_desert'] = (((FoodAccessAtlascleaned['LILATracts_halfAnd10'] == 1) | (FoodAccessAtlascleaned['LILATracts_1And20'] == 1)) & (FoodAccessAtlascleaned['HUNVFlag'] == 1)).astype(int)
print(FoodAccessAtlascleaned['is_desert'].value_counts())



# Now that we've obtained the target variable for classification, these columns tend to be unnecessary


FoodAccessAtlascleaned.drop(columns=['lapop1', 'lapop1share', 'lalowi1',
       'lalowi1share', 'lakids1', 'lakids1share', 'laseniors1',
       'laseniors1share', 'lawhite1', 'lawhite1share', 'lablack1',
       'lablack1share', 'laasian1', 'laasian1share', 'lanhopi1',
       'lanhopi1share', 'laaian1', 'laaian1share', 'laomultir1',
       'laomultir1share', 'lahisp1', 'lahisp1share', 'lahunv1', 'lahunv1share',
       'lasnap1', 'lasnap1share', 'TractLOWI'], inplace=True)


other_drops = [
    'LILATracts_1And10', 'LILATracts_1And20',
    'LA1and10', 'LA1and20', 
    'LATracts1', 'LATracts10', 'LATracts20', 
    'LAPOP1_10', 'LAPOP1_20',
    'LALOWI1_10', 'LALOWI1_20']
FoodAccessAtlascleaned.drop(columns=other_drops, inplace=True)

geo_df = gpd.read_file("../DATA/cb_2023_us_tract_5m/cb_2023_us_tract_5m.shp",)
geo_df.head()
geo_df.crs

FoodAccessAtlascleaned["CensusTract"] = FoodAccessAtlascleaned["CensusTract"].astype(str)

merged_df = FoodAccessAtlascleaned.merge(geo_df, left_on='CensusTract', right_on='GEOID', how='left')

binary_target = FoodAccessAtlascleaned['is_desert'].value_counts().to_frame().reset_index()
binary_target.columns = ['is_desert', 'count']  

melt_bin_tar = pd.melt(binary_target, id_vars='is_desert', value_vars='count')
melt_bin_tar.replace({0: 'No', 1: 'Yes'}, inplace=True)

# Exploratory Data Analysis and Data Visualization 

sns.set_theme('poster')
plt.figure(figsize=(8,9))
sns.barplot(x=melt_bin_tar['is_desert'], 
            y=melt_bin_tar['value'], 
            alpha = .80)
plt.title('Flag for Food Desert at .5 mile')
plt.ylabel('# Census Tracts')
plt.xlabel('')

county_yes = FoodAccessAtlascleaned['County'].loc[FoodAccessAtlascleaned['is_desert'] == 1].value_counts().to_frame()
county_no = FoodAccessAtlascleaned['County'].loc[FoodAccessAtlascleaned['is_desert'] == 0].value_counts().to_frame()

df = FoodAccessAtlascleaned.copy()

county_yes.columns = ['food_desert_yes']
county_no.columns = ['food_desert_no']

county_df = county_yes.join(county_no, how='outer')


county_df['Rate'] = round((county_df['food_desert_yes'] / 
                     (county_df['food_desert_yes'] + county_df['food_desert_no'])) * 100,2)


top10_rate = county_df.sort_values(by='Rate', ascending=False, inplace=False).reset_index().head(10)
county_df.replace(np.nan, 0, inplace=True)
county_df.sort_values(by='Rate', ascending=False, inplace=True)
top10_rate.plot(kind='barh',x='County', y='Rate', figsize=(12,8), color='skyblue', alpha=0.75)


top10_count = county_df.sort_values(by='food_desert_yes', ascending=False, inplace=False).reset_index().head(10)
top10_count.plot(x='County', y='food_desert_yes', kind='barh', figsize=(10, 6), color='skyblue')


gdf_tar = gpd.GeoDataFrame(merged_df[['is_desert', 'geometry']], geometry='geometry')
gdf_tar.replace({0: 'No', 1: 'Yes'}, inplace=True)


# Target Visualization on World Map

plt.rcParams.update({'legend.handletextpad':.5,
                    'legend.labelspacing':1,
                    'legend.markerscale':5,
                    'legend.fontsize':75,
                    'legend.frameon':False})


fig, ax = plt.subplots(figsize=(50, 40))


ax = gdf_tar.plot(column='is_desert',
                legend=True,
                ax=ax,
                edgecolor='face',
                alpha=0.5)


ctx.add_basemap(ax, zoom=5, url=ctx.providers.CartoDB.Positron)


leg = ax.get_legend()
leg.set_bbox_to_anchor((.20,.95))

ax.set_title(label="NYC Census Tract Flag Food Desert .5 Mile", fontdict={'fontsize': 100}, loc='center')

ax.set_axis_off()
plt.show();

sns.set_theme('notebook') 
cat_data(df)

sns.set_theme('notebook')
sns.set_style('whitegrid')

df_multi = df.select_dtypes(exclude='object')

corr = df_multi[[col for col in df_multi.columns if col != 'County']].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(21, 18))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

ax = sns.heatmap(corr, 
                 mask=mask, 
                 cmap=cmap, 
                 vmax=1, 
                 vmin=-1, 
                 center=0, 
                 square=True, 
                 linewidths=1, 
                 cbar_kws={"shrink": .75}, 
                 annot=False)



white_yes = df['lawhitehalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
white_no = df['lawhitehalfshare'].loc[df['is_desert'] == 0].to_frame().mean()
black_yes = df['lablackhalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
black_no = df['lablackhalfshare'].loc[df['is_desert'] == 0].to_frame().mean()
latino_yes = df['lahisphalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
latino_no = df['lahisphalfshare'].loc[df['is_desert'] == 0].to_frame().mean()
asian_yes = df['laasianhalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
asian_no = df['laasianhalfshare'].loc[df['is_desert'] == 0].to_frame().mean()
native_yes = df['laaianhalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
native_no = df['laaianhalfshare'].loc[df['is_desert'] == 0].to_frame().mean()
multi_yes = df['laomultirhalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
multi_no = df['laomultirhalfshare'].loc[df['is_desert'] == 0].to_frame().mean()
pacific_yes = df['lanhopihalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
pacific_no = df['lanhopihalfshare'].loc[df['is_desert'] == 0].to_frame().mean()

race_share_df = pd.DataFrame({'food_desert_yes':[white_yes['lawhitehalfshare'], 
                                                 black_yes['lablackhalfshare'],
                                                 latino_yes['lahisphalfshare'],
                                                 asian_yes['laasianhalfshare'],
                                                 native_yes['laaianhalfshare'],
                                                 multi_yes['laomultirhalfshare'],
                                                 pacific_yes['lanhopihalfshare']],
                              'food_desert_no':[white_no['lawhitehalfshare'], 
                                                black_no['lablackhalfshare'],
                                                latino_no['lahisphalfshare'],
                                                asian_no['laasianhalfshare'],
                                                native_no['laaianhalfshare'],
                                                multi_no['laomultirhalfshare'],
                                                pacific_no['lanhopihalfshare']]},
                            index=['White', 'Black', 'Latino', 'Asian', 'Native', 'Multi', 'Pacific'])



race_share_df.index.name = 'Race'
race_share_df.reset_index(inplace=True)

race_share_melted = pd.melt(
    race_share_df,
    id_vars='Race',
    value_vars=['food_desert_yes', 'food_desert_no'],
    var_name='Desert_Status',
    value_name='Share'
)

sns.set_theme('poster')
sns.catplot(x='Desert_Status', 
            y='Share', 
            hue='Race', 
            data=race_share_melted, 
            kind='bar', 
            orient='v',
            legend_out=False,
            height=12, 
            aspect=1)\
    .set(ylabel='Percent', xlabel='', title='Demographics of Low Access Pop per Census Tract')\
    .set_xticklabels(['Food Desert Yes', 'Food Desert No'])


df.groupby('County')['PovertyRate'].mean().sort_values(ascending=False).head(10)
round(df.groupby('County')['MedianFamilyIncome'].median()).sort_values(ascending=False).head(10)

income_yes = df['MedianFamilyIncome'].loc[df['is_desert'] == 1].to_frame().mean()
income_no = df['MedianFamilyIncome'].loc[df['is_desert'] == 0].to_frame().mean()

income_df = pd.concat([income_no, income_yes], axis=1)

income_df.reset_index(inplace=True)
income_df.rename({1:'Yes Food Desert',0:'No Food Desert'}, inplace=True, axis=1)

income_df.set_index('index', inplace=True)

sns.set_theme('poster')
plt.figure(figsize=(10,6))
sns.barplot(data=income_df)
plt.title('Average Median Family Income')

plt.show();

def lmplot(data, x, y, xlabel, ylabel, title, height=12, aspect=1, theme='poster', target='LILATracts_halfAnd10',\
          style='darkgrid'):
    '''Creates lmplot to comepare two variables vs the target 
    Enter dataframe, x, y, xlabel, ylabel, title.
    Height and aspect have default values
    Seaborn theme default poster, theme to darkgrid
    Target default to LILATracts_halfAnd10'''
    sns.set_style(style)
    sns.set_theme(theme)
    sns.lmplot(x=x, 
               y=y,  
               data=data,
              height=20,
              aspect=1,
               legend_out=False,
              hue=target)\
        .set(ylabel=ylabel, 
             xlabel=xlabel, 
             title=title)\
        ._legend.set_title('Target')
    plt.show();

lmplot(df, 'MedianFamilyIncome', 'PovertyRate', 'Median Income ($USD)', 'Poverty Rate (%)',\
      'Median Family Income vs Poverty Rate')


hunv_yes = df['lahunvhalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
hunv_no = df['lahunvhalfshare'].loc[df['is_desert'] == 0].to_frame().mean()

hunv_df = pd.concat([hunv_no, hunv_yes], axis=1)

hunv_df.rename({0:'Food Desert No',
               1:'Food Desert Yes'},
              inplace=True,
              axis=1)

sns.set_theme('poster')
plt.figure(figsize=(10,6))
sns.barplot(data=hunv_df)
plt.title('Percent Average Households w/o Vehicle')
plt.show();

snap_yes = df['lasnaphalfshare'].loc[df['is_desert'] == 1].to_frame().mean()
snap_no = df['lasnaphalfshare'].loc[df['is_desert'] == 0].to_frame().mean()

snap_df = pd.concat([snap_no, snap_yes], axis=1)

snap_df.rename({0:'Food Desert No',
               1:'Food Desert Yes'},
              inplace=True,
              axis=1)

sns.set_theme('poster')
plt.figure(figsize=(10,6))
sns.barplot(data=snap_df)
plt.title('Percent Average SNAP')
plt.show();



# Data Modeling 

df = df.reset_index()
df.drop(['CensusTract','State', 'County'], inplace=True, axis=1)

X = df.drop(['Urban','GroupQuartersFlag','LILATracts_halfAnd10','LILATracts_Vehicle',
             'HUNVFlag','LowIncomeTracts','LAhalfand10','LATracts_half','LATractsVehicle_20', 'is_desert'], axis=1)
y = df['is_desert']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression

lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)

y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

def get_model_scores(model, y_train, y_train_pred, y_test, y_test_pred, ):
    '''Enter model type, y_train, y_train_pred, y_test, y_test_pred.
    Returns modeling scores printed out in a consistent format'''
    print('\033[1m' + 'Below are the scoring metrics for {} model'.format(model) + '\033[0m')
    print('\n')
    print('Train {} Accuracy: {}'.format(model, metrics.accuracy_score(y_train, y_train_pred)))
    print('Test {} Accuracy: {}'.format(model, metrics.accuracy_score(y_test, y_test_pred)))
    print('\n')
    print('Train {} F1: {}'.format(model, metrics.f1_score(y_train,y_train_pred)))
    print('Test {} F1: {}'.format(model, metrics.f1_score(y_test,y_test_pred)))
    print('\n')
    print('Train {} Recall: {}'.format(model, metrics.recall_score(y_train, y_train_pred)))
    print('Test {} Recall: {}'.format(model, metrics.recall_score(y_test, y_test_pred)))
    print('\n')
    print('Train {} Precision: {}'.format(model, metrics.precision_score(y_train, y_train_pred)))
    print('Test {} Precision: {}'.format(model, metrics.precision_score(y_test, y_test_pred)))

get_model_scores('Logistic Regression', y_train, y_train_pred_lr, y_test, y_test_pred_lr)

confusion_matrix(lr, X_test, y_test, 'Logistic Regression Confusion Matrix', normalize=None)

coef=list(lr.coef_[0])
col=X.columns
final_coef={}
for i in range(len(coef)):
    final_coef[col[i]]=coef[i]
final_coef=pd.DataFrame(final_coef,index=[0]).T
final_coef.columns=['Coeficients']
final_coef=final_coef.sort_values(ascending=False,by='Coeficients')
print(final_coef.head(5))
print(final_coef.tail(5))


selector = RFECV(estimator=lr, step=1, cv=5, scoring='f1', n_jobs=-1)
selector.fit(X_train, y_train)
rfe_features = X.columns[(selector.get_support())]

X_rfe = df[rfe_features]
Y = y.copy()

lr2=LogisticRegression(class_weight="balanced")
X_train2,X_test2,y_train2,y_test2=train_test_split(X_rfe,Y,random_state=2020, test_size=.2)

scaler = MinMaxScaler()
X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.transform(X_test2)

lr2.fit(X_train2,y_train2)

y_train_pred_lr2 = lr2.predict(X_train2)
y_test_pred_lr2 = lr2.predict(X_test2)

get_model_scores('Logistic Regression w/RFE', y_train2, y_train_pred_lr2, y_test2, y_test_pred_lr2)

confusion_matrix(lr2, X_test2, y_test2, 'Logistic Regression w/ RFE Confusion Matrix', normalize=None)

# Decision Tree

dtc = DecisionTreeClassifier(criterion='gini', max_depth=5, class_weight='balanced', random_state=1)

dtc = dtc.fit(X_train, y_train)

y_train_pred_dtc = dtc.predict(X_train)
y_test_pred_dtc  = dtc.predict(X_test)

get_model_scores('Decision Tree', y_train, y_train_pred_dtc, y_test, y_test_pred_dtc)
confusion_matrix(dtc, X_test, y_test, 'Decision Tree Classifier Confusion Matrix', normalize=None)


# Random Forest

rf = RandomForestClassifier(random_state = 1,
                            class_weight='balanced')

rf.fit(X_train, y_train)

y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

get_model_scores("Random Forest", y_train, y_train_pred_rf, y_test, y_test_pred_rf)

confusion_matrix(rf, X_test, y_test, "Random Forest", normalize=None)

# XG Boost

params = {'objective':'binary:logistic',
        'colsample_bytree':0.5,
        'subsample':0.5,
        'learning_rate':0.1,
        'max_depth':4,
        'alpha':1,
        'n_estimators':130,
        'scale_pos_weight':68.032258}

xg = XGBClassifier(**params)


X_train = pd.DataFrame(X_train,columns=X.columns)
X_test = pd.DataFrame(X_test,columns=X.columns)


xg.fit(X_train, y_train)


y_train_pred_xg = xg.predict(X_train)
y_test_pred_xg = xg.predict(X_test)

get_model_scores('XG Boost', y_train, y_train_pred_xg, y_test, y_test_pred_xg)

confusion_matrix(xg, X_test, y_test, 'XG Boost Confusion Matrix', normalize=None)

xg_grid_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': range(1,5,1)
    }

xg_grid = XGBClassifier(scale_pos_weight = 68.032258, random_state=1)

grid_xg = GridSearchCV(estimator=xg_grid,
                      param_grid=xg_grid_params,
                      cv=5,
                      n_jobs=-1,
                      verbose=1)

grid_xg.fit(X_train, y_train)

y_train_pred_grid_xg = grid_xg.predict(X_train)
y_test_pred_grid_xg = grid_xg.predict(X_test)

get_model_scores("XGBoost w/ GridSearch", y_train, y_train_pred_grid_xg, y_test, y_test_pred_grid_xg)

grid_xg.best_params_

confusion_matrix(grid_xg, X_test, y_test, "XGBoost w/ GridSearch", normalize=None)


