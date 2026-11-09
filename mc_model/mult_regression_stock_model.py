# %%
# Pulls in input and comparison data from excel files
def inputs():
    inputs_spy_df = 
    inputs_cpi_df = 
    inputs_fed_df = 
    inputs_oil_df = 
    inputs_ppi_df = 
    inputs_unr_df = 

    return inputs_spy_df, inputs_cpi_df, inputs_fed_df, inputs_oil_df, inputs_ppi_df, inputs_unr_df

# %% 
# Load in the multiple linear regression model
def model_load():
    filename - r''
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

# %%
# This function sets the start date for each input dataframe equal to each other
# This start date is the start date of the spy dataframe
def start_date_for_each_df_besides_spy(oil_df, cpi_df, ppi_df, unr_df, fed_df):

    start_date = '1993-02-01'
    oil_df, cpi_df, ppi_df, unr_df, fed_df = (df.loc[(df['DATE'] >= start_date)] for df in (oil_df, cpi_df, ppi_df, unr_df, fed_df))

    return oil_df, cpi_df, ppi_df, unr_df, fed_df

# %% 
# This function ensures that we only pull in the last full month of data for the spy dataframe
def end_date_for_spy_df(input_df):

    input_df['Date'] = pd.to_datetime(input_df['Date'])

    spy_max_date = input_df['Date'].max()
    second_to_last_month = spy_max_date - pd.DateOffset(months=1)

    spy_df = input_df[input_df['Date'] < second_to_last_month]

    return spy_df

# %%
# This function merges all of the dataframes into a single dataframe
def merge_all_dfs_for_predictions(oil_df, cpi_df, ppi_df, unr_df, fed_df, spy_df):

    df = oil_df.merge(cpi_df, on='DATE')
    df = df.merge(ppi_df, on='DATE')
    df = df.merge(fed_df, on='DATE')
    df = df.merge(unr_df, on='DATE')

    spy_df = spy_df[['DATE', 'Adj Close']]

    predictions_merged_df = df.merge(spy_df, on='DATE')

    saved_df = predictions_merged_df
    filename = 'predictions_merged_df.pkl'
    pickle.dump(saved_df, open(filename, 'wb'))

    return predictions_merged_df


# %%
# This function performs some final preprocessing on teh spy dataframe and saves the final merged dataframe to a .pkl file
def preprocessing_final(df):

    df['stock_price_diff'] = df['Adj Close'].diff()
    df['stock_price_diff'] = df['stock_price_diff'].fillna()

    saved_df = df
    filename = 'merged_df.pkl'
    pickle.dump(saved_df, open(filename, 'wb'))

    return df


# %%
# Train test split
def set_up_train_test_split(df):

    x = df[['POILWTIUSDM', 'CPIAUCSL', 'PPIACO', 'FEDFUNDS', 'UNRATE']]
    y = df['Adj Close']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2025)

    return x_train, x_test, y_train, y_test, x, y

# %%
# Multiple Linear Regression
def fit_and_evaluate_model_performance(x_train, y_train, x, y, x_test, y_test):

    lr = LinearRegression()
    lrmodel = lr.fit(x_train, y_train)

    print("Intercept: ", lr.intercept_)
    print("Coefficients: ", list(zip(x, lr.coef_)))
    coef_df = pd.DataFrame({'Variable': x, 'Coefficient': lr.coef_})
    coef_df = coef_df.transpose()
    coef_df = coef_df.set_axis(coef_df.iloc[0], axis=1, inplace = False)
    coef_df = coef_df[1:]
    coef_df = coef_df.reset_index(drop=True)
    intercept_df = pd.DataFrame({'Intercept': coef_df.index, 'Value': lr.intercept_})
    intercept_df = intercept_df.drop(intercept_df.columns[0], axis=1)
    model_df = pd.merge(coef_df, intercept_df, left_index=True, right_index=True)

    y_pred = lr.predict(x_test)

    y_train_pred = lr.predict(x_train)

    result_df = pd.DataFrame({'ACtual': y_test, 'Predicted': y_pred})

    meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
    meanSqErr = metrics.mean_squared_error(y_test, y_pred)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('R squared: {:.2f}'.format(lr.score(x,y)*100))
    print('Mean Absolute Error: ', meanAbErr)
    print('Mean Square Error: ', meanSqErr)
    print('Root Mean Square Error: ', rootMeanSqErr)

    print('RMSE training data = ', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    print("RMSE test data = ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    stat, p = pearsonr(y_test, y_pred)

    print('correlation')
    print(stat)

    plt.scatter(x = result_df.index, y = result_df['Actual'], label = 'Actual')
    plt.scatter(x = result_df.index, y = result_df['Predicted'], label = 'Predicted')
    plt.legend()

    return result_df, model_df, lrmodel 




    
# %%
# Only predict months in 2024 given inputs from 2024