# Intraday_Electric_Price_Forecast
This project aims to analyze the factors influencing electricity prices and assess the feasibility of intra-day forecasts for the Spanish electricity grid. By using a dataset that includes various load, price, and weather features, I try  to identify the most relevant predictors that can enhance the accuracy of electricity price forecasting.

Dataset Description

The dataset comprises a range of features that capture the dynamics of electricity generation, consumption, and weather conditions. Below is a detailed description of each predictor included in the dataset:
Time

    Format: Timestamp in the format: YYYY-MM-DD HH:MM:SS±HH:MM
    Description: Represents the specific date and time for each data entry, allowing for temporal analysis of electricity prices and generation.
Generation Features

    Generation Biomass: Power generation from biomass sources (MW).
    Generation Fossil Brown Coal/Lignite: Power generation from brown coal or lignite (MW).
    Generation Fossil Coal-Derived Gas: Power generation from gas produced from coal (MW).
    Generation Fossil Gas: Power generation from fossil gas (MW).
    Generation Fossil Hard Coal: Power generation from hard coal (MW).
    Generation Fossil Oil: Power generation from fossil oil (MW).
    Generation Fossil Oil Shale: Power generation from oil shale sources (MW).
    Generation Fossil Peat: Power generation from peat sources (MW).
    Generation Geothermal: Power generation from geothermal energy (MW).
    Generation Hydro Pumped Storage Consumption: Power consumption from pumped storage hydroelectric systems (MW).
    Generation Hydro Pumped Storage Aggregated: Total power generation from aggregated pumped storage hydroelectric systems (MW).
    Generation Hydro Run-of-River and Poundage: Power generation from run-of-river hydroelectric systems (MW).
    Generation Hydro Water Reservoir: Power generation from hydroelectric systems with water reservoirs (MW).
    Generation Marine: Power generation from marine energy sources (MW).
    Generation Nuclear: Power generation from nuclear energy (MW).
    Generation Other: Power generation from other unspecified sources (MW).
    Generation Other Renewable: Power generation from other renewable sources (MW).
    Generation Solar: Power generation from solar energy (MW).
    Generation Waste: Power generation from waste materials (MW).
    Generation Wind Onshore: Power generation from onshore wind turbines (MW).
    Generation Wind Offshore: Power generation from offshore wind turbines (MW).

Forecast Features

    Forecast Solar Day Ahead: Forecast of solar power generation for the next day (MW).
    Forecast Wind Onshore Day Ahead: Forecast of onshore wind power generation for the next day (MW).
    Forecast Wind Offshore Day Ahead: Forecast of offshore wind power generation for the next day (MW).
    Total Load Forecast: Forecast of total power load (MW).

Actual Load Features

    Total Load Actual: Actual total power load (MW).

Price Features

    Price Day-Ahead: Forecasted Day-Ahead price of electricity (€/MWh).
    Price Actual: Current electricity price (€/MWh).

Weather Features

    city_name: Name of the city for which the weather data is provided (options: Valencia, Madrid, Bilbao, Barcelona, or Seville).
    temp: Current temperature in Kelvin (K).
    temp_min: Minimum temperature recorded for the day in Kelvin (K).
    temp_max: Maximum temperature recorded for the day in Kelvin (K).
    pressure: Atmospheric pressure at sea level measured in hPa (hectopascals).
    humidity: Humidity level expressed as a percentage (%).
    wind_speed: Wind speed measured in meters per second (m/s).
    wind_deg: Wind direction indicated in degrees (°) from true north.
    rain_1h: Amount of rain that has fallen in the last hour, measured in millimeters (mm).
    rain_3h: Amount of rain that has fallen in the last three hours, measured in millimeters (mm).
    snow_3h: Amount of snow that has fallen in the last three hours, measured in millimeters (mm).
    clouds_all: Cloud cover percentage indicating how much of the sky is covered by clouds (%).
    weather_id: Weather condition code representing the current weather status.
    weather_main: A short description of the current weather
    weather_description: A long description of the current weather
    weather_icon: Weather icon code used by the Open Weather API

The primary objectives of this project are to:

    Identify key features that impact electricity prices in the Spanish grid.
    Develop and validate models for intra-day forecasting of electricity prices using the identified predictors.
    Provide insights into the relationship between weather, generation, load, and electricity pricing.
