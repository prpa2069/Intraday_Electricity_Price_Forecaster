import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from datetime import timedelta
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import plotly.graph_objs as go

#streamlit run .\electricity_price_predictor_app



# Load datasets
def load_dataset(path):
    # Check the file extension to determine the appropriate reading method
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")
    
    # Convert the 'time' column to datetime if it exists
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    
    return df

def manipulate_datasets(energy_df, weather_df):
    #Energy Features
    energy_df["time"] = pd.to_datetime(energy_df["time"], utc = True)
    weather_df["time"] = pd.to_datetime(weather_df["time"], utc = True)
    energy_generation_columns = ['generation biomass', 'generation fossil brown coal/lignite',
       'generation fossil gas', 'generation fossil hard coal',
       'generation fossil oil', 'generation hydro pumped storage consumption',
       'generation hydro run-of-river and poundage',
       'generation hydro water reservoir', 'generation nuclear',
       'generation other', 'generation other renewable', 'generation solar',
       'generation waste', 'generation wind onshore'
]
    energy_df["total_gen"] = energy_df[energy_generation_columns].sum(axis=1)

    #Weather data features creation
    weather_df["temp_diff"] = weather_df["temp_max"] - weather_df["temp_min"]
    weather_df.drop(columns=["temp_max","temp_min"],inplace=True)
    #Combine the dfs
    cities = weather_df["city_name"].unique()
    user_data = energy_df.copy()
    for city in cities:
        #indexes values for a certain city name and drops the column
        df_city = weather_df[weather_df["city_name"] == city]
        df_city = df_city.drop(columns="city_name")
        
        #Duplicate times still exist within the weather dataset with a unique city. To gain a unique time, a 
        numeric_cols = df_city.select_dtypes(include=['number']).columns
        
        #Assigns the mean of duplicate time value entries to a unique time value
        df_city = df_city.groupby('time')[numeric_cols].agg('mean').reset_index()
        
        #Rename columns to include the city name
        for col in numeric_cols:
            df_city = df_city.rename(columns={col: col + "_" + city})
        
        #Merge the dfs
        user_data = user_data.merge(df_city, on="time", how="inner")
    #Drop columns as no correlation exists as noted in data exploration
    try:
        user_data.drop(['snow_3h_ Barcelona', 'snow_3h_Seville'], axis=1, inplace=True)
    except:
        raise st.error(f"Ensure all five cities are represented within the dataset")
    #Fill nan values with mean
    user_data = user_data.fillna(user_data.mean())

    return user_data

def create_engineered_features(df, feature_engineered_data): 
    #Time Feats
    warnings.filterwarnings("ignore")
    def create_time_features(df, create_time_period_feature = False, step = 6):
        #Creates time related features of month, day, hour
        time_feat_df = pd.DataFrame()
        time_feat_df = df[["time","price actual"]]

        time_feat_df["month"] = time_feat_df['time'].dt.month_name()
        time_feat_df["day"] = time_feat_df['time'].dt.day_name()
        time_feat_df["hour"] = time_feat_df["time"].dt.hour.astype('category')
        #Creates stepped features based on hour
        #using pd.getdummies creates 23 additional features, whereas hour features could be captured via a period reducing dimensionality
        if create_time_period_feature:
            for i in range(step, 24, step):
                time_feat_df[f"is_btwn_{i}_and_{i+step}"] = time_feat_df["hour"].isin(range(i, i + step)).astype(bool)

            time_feat_df.drop(columns=["hour"],inplace=True)
        
        return time_feat_df
    time_best_feat = create_time_features(df, create_time_period_feature=True,step = 6).drop(columns="price actual")

#Lag and MA feats
# Create time features
    lag_ma_feats = create_time_features(df)
    lag_ma_feats["year"] = lag_ma_feats["time"].dt.year
    lag_ma_feats["week"] = lag_ma_feats["time"].dt.isocalendar().week
    lag_ma_feats.set_index("time", inplace=True)

    # Compute weekly average and merge with previous years for comparison
   
    weekly_avg_last_year = lag_ma_feats.groupby(["year", "week"])["price actual"].mean().reset_index()
    weekly_avg_last_year["last_year_weekly_mean_price"] = weekly_avg_last_year.groupby("week")["price actual"].shift(1)
    # Determine if a "last year exists within data, if it does not fill with mean from df_train"   
    if df.loc[df.index[-1], "time"] - timedelta(days=365) not in df["time"].values:
        weekly_avg_last_year["last_year_weekly_mean_price"].fillna(feature_engineered_data.groupby(feature_engineered_data["time"].dt.isocalendar().week)["last_year_weekly_mean_price"].transform("mean"), inplace=True)
    else:
        weekly_avg_last_year["last_year_weekly_mean_price"].fillna(weekly_avg_last_year.groupby("week")["last_year_weekly_mean_price"].transform("mean"), inplace=True)

    # Create lagged features
    for lag in [24, 24*7]: 
        lag_ma_feats[f"prev_{'day' if lag == 24 else 'week'}_price"] = lag_ma_feats["price actual"].shift(lag)
        lag_ma_feats[f"prev_{'day' if lag == 24 else 'week'}_price"].fillna(lag_ma_feats["prev_day_price"].mean(), inplace=True)

    # Create moving averages
    for period in [1, 3, 7, 14, 30, 90]:
        lag_ma_feats[f"{period}_day_ma"] = lag_ma_feats["price actual"].rolling(f"{period}D", min_periods=1).mean()

    lag_ma_feats = lag_ma_feats.reset_index().merge(weekly_avg_last_year.drop(columns="price actual"), on=["year", "week"], how="outer")
    lag_ma_feats.drop(columns=['month', 'day','hour','year', 'week'], inplace=True)
    lag_ma_feats.drop(columns="price actual",inplace=True)
    #Create a combined dataset
    merged_time_lag_ma_feat = pd.merge(time_best_feat,lag_ma_feats,on="time",how="inner")

    df = df.merge(merged_time_lag_ma_feat,on="time",how="inner")
    
    #Create quadratic trend features
    df.reset_index(inplace=True)
    df.rename(columns={"index":"t"},inplace=True)
    df["t^2"] = df["t"]**2
    # Create dummy variablse for non-numeric values while avoid co-linearity
    df = pd.get_dummies(df,drop_first=True)
    return df

def apply_transformations(user_data, train_df):
    #Int transformers
    transformer_scaler = joblib.load(r"transformers\scaler.pkl")
    transformer_pca = joblib.load(r"transformers\pca.pkl")

    #Create variables for storing time, response variable, and predictors
    time = user_data["time"]
    y = user_data["price actual"]
    user_data.drop(columns = ["time","price actual"], inplace = True)

    #Encode the variables via dummies
    user_data = pd.get_dummies(user_data, drop_first=True)
    #Interlay missing features based off train_df
    user_data = user_data.reindex(columns=train_df.columns, fill_value=False)

    #Transform the columns
    # For user data, scale numeric columns
    numeric_cols_val = user_data.select_dtypes(include=['float64', 'int64']).columns
    boolean_cols_train = user_data.select_dtypes(include=['bool']).columns

    scaled_numeric_val = transformer_scaler.transform(user_data[numeric_cols_val])

    # Create a DataFrame with the scaled numeric data for user data
    scaled_numeric_val_df = pd.DataFrame(scaled_numeric_val, columns=numeric_cols_val)

    # Combine the scaled numeric DataFrame with the original boolean DataFrame for user data
    user_data = pd.concat([scaled_numeric_val_df, user_data[boolean_cols_train].reset_index(drop=True)], axis=1)

    #Apply PCA transform
    user_data_pca_X = transformer_pca.transform(user_data)

    #Converts the pca_data into a dataframe, similar to what was passed for easy manipulation.
    columns = [f"Column {i+1}" for i in range(user_data_pca_X.shape[1])]
    user_data_pca_X = pd.DataFrame(user_data_pca_X, columns=columns)
    pca_data_df = pd.concat([time,user_data_pca_X,y],axis=1)
    return pca_data_df
        
def upload_file(label):
    """Function to upload a CSV file and return the DataFrame."""
    uploaded_file = st.file_uploader(label, type=["csv"], accept_multiple_files=False)
    if uploaded_file is not None:
        try:
            # Read the CSV file
            data = pd.read_csv(uploaded_file)
            # Check if the DataFrame is empty
            if data.empty:
                st.warning("The uploaded file is empty. Please upload a valid CSV file.")
                return None
            else:
                st.write("Partial data from file:")
                st.dataframe(data)
                return data
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            return None
    return None

def create_sidebar():
    #Create a header and add a graphic
    st.sidebar.header("Power Generation Features")
    st.sidebar.image(Image.open(r'images\background\stonks_rising.jpg'), use_container_width =True)
    # Define the features and their descriptions
    features = {
        "Time": "Timestamp in the format: YYYY-MM-DD HH:MM:SS±HH:MM ",
        "Generation Biomass": "Power generation from biomass sources (MW).",
        "Generation Fossil Brown Coal/Lignite": "Power generation from brown coal or lignite (MW).",
        "Generation Fossil Gas": "Power generation from fossil gas (MW).",
        "Generation Fossil Hard Coal": "Power generation from hard coal (MW).",
        "Generation Fossil Oil": "Power generation from fossil oil (MW).",
        "Generation Hydro Pumped Storage Consumption": "Power consumption from pumped storage hydroelectric systems (MW).",
        "Generation Hydro Run-of-River and Poundage": "Power generation from run-of-river hydroelectric systems (MW).",
        "Generation Hydro Water Reservoir": "Power generation from hydroelectric systems with water reservoirs (MW).",
        "Generation Nuclear": "Power generation from nuclear energy (MW).",
        "Generation Other": "Power generation from other sources not specified (MW).",
        "Generation Other Renewable": "Power generation from other renewable sources (MW).",
        "Generation Solar": "Power generation from solar energy (MW).",
        "Generation Waste": "Power generation from waste materials (MW).",
        "Generation Wind Onshore": "Power generation from onshore wind turbines (MW).",
        "Forecast Solar Day Ahead": "Forecast of solar power generation for the next day (MW).",
        "Forecast Wind Onshore Day Ahead": "Forecast of onshore wind power generation for the next day (MW).",
        "Total Load Forecast": "Forecast of total power load (MW).",
        "Total Load Actual": "Actual total power load (MW).",
        "Price Day-Ahead": "Forecasted Day-Ahead price of electricity (€/MWh)",
        "Price Actual": "Current electricity price (€/MWh)",
        "city_name": "Name of the city for which the weather data is provided. Either Valencia, Madrid, Bilbao, Barcelona, or Seville",
        "temp": "Current temperature in Kelvin (K).",
        "temp_min": "Minimum temperature recorded for the day in Kelvin (K).",
        "temp_max": "Maximum temperature recorded for the day in Kelvin (K).",
        "pressure": "Atmospheric pressure at sea level measured in hPa (hectopascals).",
        "humidity": "Humidity level expressed as a percentage (%).",
        "wind_speed": "Wind speed measured in meters per second (m/s).",
        "wind_deg": "Wind direction indicated in degrees (°) from true north.",
        "rain_1h": "Amount of rain that has fallen in the last hour, measured in millimeters (mm).",
        "rain_3h": "Amount of rain that has fallen in the last three hours, measured in millimeters (mm).",
        "snow_3h": "Amount of snow that has fallen in the last three hours, measured in millimeters (mm).",
        "clouds_all": "Cloud cover percentage indicating how much of the sky is covered by clouds (%).",
        "weather_id": "Weather condition code representing the current weather status."
    }



    # Write the descriptions of the features in the sidebar
    for feature, description in features.items():
        st.sidebar.write(f"**{feature}:** {description}")



#Create two columns
st.set_page_config(layout="wide")
#Generate a sidebar that provides a basic description of features
create_sidebar()

#Title and graphic
st.markdown("<h1 style='text-align: center;'>Gooder Spain Energy Price Predictor</h1>", unsafe_allow_html=True)
st.image(Image.open(r'images\background\spain_beach.jpg'), use_container_width =True)

#Intialize variables based on previous stored data
weather_df = pd.read_csv(r"engineered_data\weather_data_val_manipulated.csv")
energy_df = pd.read_csv(r"engineered_data\energy_data_val_manipulated.csv")
weather_id_desc = pd.read_csv(r"engineered_data\weather_id_descriptors.csv")
engineered_features_data = load_dataset(r"engineered_data\feature_engineered_data.csv")
ml_data_X_train = load_dataset(r"engineered_data\feature_engineered_data_X_train.parquet")
best_model_linear_reg = joblib.load(r"models/best_model_linear_reg.pkl")

#Output examples to provide a tabular representation of features
st.write("Example weather data:")
st.dataframe(weather_df[weather_df["time"] == weather_df.loc[0, "time"]], use_container_width=True, hide_index=True)

st.write("Example energy data:")
st.dataframe(energy_df[:5], use_container_width=True, hide_index=True)

st.write("Weather ID Unique Identifiers:")
st.dataframe(weather_id_desc, use_container_width=True, hide_index=True)

# Split layout into two columns
right, left = st.columns(2)









# Right column - Prediction Button
with right:
    # User input for weather and energy data
    st.subheader("Data Upload")
    st.write("Please ensure that the times match and weather data for Valencia, Madrid, Bilbao, Barcelona, Seville are provided")
    weather_user_data = upload_file("Please upload a CSV file with the Spain Weather predictors.")
    energy_user_data = upload_file("Please upload a CSV file with the Spain Energy predictors.")

    # User input validation
    if weather_user_data is not None and energy_user_data is not None:
        if "price actual" not in energy_user_data or "temp" not in weather_user_data:
            st.error("Improper dataset uploaded. Ensure the correct datasets")
        elif len(weather_user_data) < 168:
            st.error("Please provide at least 7 days of data")
        else:
            st.success("Both datasets uploaded successfully!")
        if len(energy_user_data) != len(weather_user_data)/5:
            st.warning("Amount of unique timestamps for each city does not match the length of the energy dataset. Performance of model may be affected")
        

    #Button press for prediction
    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: rgb(204, 49, 49);
        }
        </style>""", unsafe_allow_html=True)
    
    b = st.button("Predict")
    #When button is pressed == True
    if b:
        #If no proper data is provided default to a pre-defined dataset
        if weather_user_data is None or energy_user_data is None:
            st.write("A portion of the original datasets will be used as no/improper data has been provided for either weather or energy")
            weather_user_data = weather_df
            energy_user_data = energy_df
        # Check and add space only to 'Barcelona', original data provided the value with a space as " Barcelona"
        # Values when trained were " Barcelona". Keeping consistency with the feature names
        weather_user_data['city_name'] = weather_user_data['city_name'].apply(lambda x: " " + x if x == "Barcelona" and not x.startswith(" ") else x)
        
        #Apply transformations
        try:
            combined_user_data = manipulate_datasets(energy_user_data,weather_user_data)
            combined_user_data = create_engineered_features(combined_user_data, engineered_features_data)
            combined_user_data = apply_transformations(combined_user_data, ml_data_X_train)
        except Exception as e:
            st.error(f"An error occurred: {e} Ensure the feature names match the example") # Added the exception message for better debugging

        #Prediction
        input_pred = best_model_linear_reg.predict(combined_user_data.drop(columns=["time", "price actual"]))
        y_actual = combined_user_data["price actual"]
        
        
        st.write(f"<span style='color: blue;'>Average Price predicted by the button: €{np.mean(input_pred):,.2f}</span>", unsafe_allow_html=True)

        #Metrics
        mae = mean_absolute_error(y_actual, input_pred)
        mse = mean_squared_error(y_actual, input_pred)
        rmse = root_mean_squared_error(y_actual, input_pred)
        st.subheader("Prediction Metrics")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(
            f"""
            <span style='color: green;'>
                With a reduction in data, performance of the model is typically reduced as well. 
                This is a result of the additional lagging and moving average features 
                requiring at least two year's worth of data. If enough data is not provided,
                the lagging features will be the mean from the train dataset and moving averages
                will consist of the same value till a certain timeframe passes.
            </span>
            """,
            unsafe_allow_html=True
        )

# Left column: Graphical Representations of Data
with left:
    def calculate_weekly_averages(x, y):
        # Flatten the x array (e.g., dates) to 1D
        x = x.ravel()
        
        # Try to convert y to its raw values (e.g., if it's a DataFrame column)
        try:
            y = y.values
        except:
            pass  # If that doesn't work, just keep y as it is
        
        # Flatten y array (e.g., values) to 1D
        y = y.ravel()
        
        # Create a DataFrame using x as dates and y as values
        df = pd.DataFrame({'date': x, 'value': y})
        
        # Set the 'date' column as the index for time-based operations
        df.set_index('date', inplace=True)
        
        # Group the data by week and calculate the average value for each week
        weekly_avg = df.resample('W').mean()
        
        # Return the dates (weekly) and their corresponding average values
        return weekly_avg.index, weekly_avg['value']

    def plot_model_preds(time, y_actual, y_pred):
        # Calculate weekly averages for actual and predicted values
        x_actual_weekly, y_actual_weekly = calculate_weekly_averages(time, y_actual)
        x_pred_weekly, y_pred_weekly = calculate_weekly_averages(time, y_pred)

        # Create an interactive plot using Plotly
        fig = go.Figure()

        # Add actual values to the plot
        fig.add_trace(go.Scatter(x=x_actual_weekly, y=y_actual_weekly, mode='lines', name='Weekly Avg Actual', line=dict(color='blue')))

        # Add predicted values to the plot
        fig.add_trace(go.Scatter(x=x_pred_weekly, y=y_pred_weekly, mode='lines', name='Weekly Avg Predictions', line=dict(color='green', dash='dash')))

        # Update layout
        fig.update_layout(
            title='Weekly Averaged Actuals vs. Predictions',
            xaxis_title='Date',
            yaxis_title='Price (€/MWh)',
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )

        # Display the interactive plot in Streamlit
        st.plotly_chart(fig)

    st.markdown("<h1 style='text-align: center;'>Observed vs Predicted Values During Training</h1>", unsafe_allow_html=True)
    
    # Output results of training and testing set
    st.write("Five-Fold Time Series CV Linear Regression Model - Train vs. Val")
    st.image(Image.open('images\graphs\linear_model_weekly_actual_preds.png'), use_container_width =True)
    
    # If button is pressed, output an interactable graph for the user dataset
    if b:
        st.write("Actual vs. Prediction on provided dataset")
        #plot_model_preds(combined_user_data['time'],y_actual, input_pred)
        graph_df = combined_user_data[["time","price actual"]]
        graph_df["Predicted"] = input_pred
        #st.line_chart(graph_df.set_index('time')[['price actual', 'Predicted']])
        plot_model_preds(combined_user_data["time"], y_actual, input_pred)
