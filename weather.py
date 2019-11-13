#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:25:32 2019
@author: fhall, spell1, jhardy
"""

import pyowm
from owm_token import owm_token
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# API token
owm = pyowm.OWM(owm_token)

def legit_city(city):
    try:
        # forecast of 5 days out with 3 hour intervals
        owm.three_hours_forecast(city + "," + 'US')
        return('OK, let me find that 5-day forecast!')
    except:
        return(city + " is not a valid location. Please try again")


# WEATHER FORECAST FUNCTION #
def weather_forecast(location):

    forecast = owm.three_hours_forecast(location + "," + 'US')
     
    # Set days until arrival
    arrive = 1
    
    forecastDate = datetime.now()
    
    # Get weather
    weather_hours = []
    for i in range (0,99,3):
        forecastDate = datetime.now() + timedelta(days=arrive, hours=i)
        weather_hours.append(forecast.get_weather_at(forecastDate))
    
    temp = []
    detail = []
    dt = []
    wind = []

    for weather in weather_hours:
        description = weather.get_detailed_status()
        time = weather.get_reference_time()
        temperature = weather.get_temperature()['temp'] * 9/5 - 459.67
        wind_speed = weather.get_wind()['speed']
        t = datetime.fromtimestamp(float(time))
        # add to lists
        dt.append(t)
        temp.append(temperature)
        detail.append(description)
        wind.append(wind_speed)
        
    # Create the plot
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(dt,temp,'-p',color='dodgerblue',alpha=.9, linewidth=6)
    fig.suptitle("{}'s 5-day forecast".format(location), fontsize=22)
    plt.ylabel('Temperature (F)', fontsize=14)
    
    # Affordance for Freezing 
    plt.axhline(32, linewidth=2, color='k', linestyle='dashed')
    # format x axis
    ax.xaxis.set_major_formatter(DateFormatter("%a %H:%M"))
    # save
    plt.savefig("plots/forecast.png")
