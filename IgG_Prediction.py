import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import sys

def run_analysis(train_csv, test_csv, output_csv):
    # Load training and testing datasets
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    def evaluate_model(model, params, X_train, y_train, niter):
        if params:
            random_search = RandomizedSearchCV(model, params, n_iter=niter, cv=5, verbose=0, random_state=42, n_jobs=-1)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            best_score = best_model.score(X_train, y_train)
            
        else:
            best_model = model.fit(X_train, y_train)
        return best_model, best_score

    # Prepare the data for training
    day_0_data_train = df_train[df_train['planned_day_relative_to_boost'] == 0].set_index('subject_id')
    day_14_data_train = df_train[df_train['planned_day_relative_to_boost'] == 14].set_index('subject_id')
    day_0_data_train = day_0_data_train[day_0_data_train.index.isin(day_14_data_train.index)]
    day_14_data_train = day_14_data_train[day_14_data_train.index.isin(day_0_data_train.index)]

    # Select relevant features
    features = ['IgG_PT', 'biological_sex', 'race', 'infancy_vac', 'age']
    X_train = pd.get_dummies(day_0_data_train[features], columns=['biological_sex', 'race', 'infancy_vac'])
    y_train = day_14_data_train['IgG_PT']

    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_scaled_train)

    # Prepare the data for test
    day_0_data_test = df_test[df_test['planned_day_relative_to_boost'] == 0].set_index('subject_id')
    X_test = pd.get_dummies(day_0_data_test[features], columns=['biological_sex', 'race', 'infancy_vac'])
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]

    # Standardize and apply PCA to test data
    X_scaled_test = scaler.transform(X_test)
    X_test_pca = pca.transform(X_scaled_test)

    # Define your models and their respective parameter grids
    models = [
        (GradientBoostingRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]}, 27),
        (xgb.XGBRegressor(), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 4], 'colsample_bytree': [0.3, 0.7]}, 16),
        (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}, 10),
        (SVR(), {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf', 'poly']}, 24)
    ]

    best_score = float('-inf')
    best_model = None

    for model, params, niter in models:
        trained_model, score = evaluate_model(model, params, X_train, y_train, niter)
        if score > best_score:
            best_score = score
            best_model = trained_model

    print(f'The best model is: {best_model}, and the best score is: {best_score}')
    predictions = best_model.predict(X_test)

    # Save predictions
    test_subject_ids = day_0_data_test.reset_index()['subject_id']
    results_df = pd.DataFrame({
        'subject_id': test_subject_ids,
        'IgG_PT': predictions
    })
    results_df.drop_duplicates(inplace=True)
    day_0_data_test_unique = day_0_data_test[~day_0_data_test.index.duplicated(keep='first')]
    day_0_values = day_0_data_test_unique['IgG_PT']
    fold_change = results_df['IgG_PT'] / day_0_values.reindex(results_df['subject_id']).values
    results_df['FC_IgG_PT'] = fold_change
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    results_df.dropna(inplace=True)
    results_df['IgG_PT_rank'] = results_df['IgG_PT'].rank(ascending=False, method='min').astype(int)
    results_df['IgG_PT-FC_rank'] = results_df['FC_IgG_PT'].rank(ascending=False, method='min').astype(int)
    results_df.drop(columns=['IgG_PT', 'FC_IgG_PT'], inplace=True)
    results_df.to_csv(f'{output_csv}/igg_fc_testing.csv', index=False)
