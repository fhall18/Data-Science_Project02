import pyowm
from owm_token import owm_token
from datetime import datetime, timedelta

owm = pyowm.OWM(owm_token)
country = 'US'

# Set location
validLocation = False
while validLocation == False:
    try:
        location = 'Burlington' #input('what city or town? ')
        # forecast of 5 days out with 3 hour intervals
        forecast = owm.three_hours_forecast(location + "," + country)
        validLocation = True
    except:
        print(location, " is not a valid location.")
        continue


# Set days until arrival
arrive = 100
while arrive not in range(6):
    arrive = 1 #int(input("How many days are there before you arrive (0-5)? "))

forecastDate = datetime.now() + timedelta(days=arrive, hours=6)

# Get weather
weather_hours = []
for i in range (0,25,3):
    forecastDate = datetime.now() + timedelta(days=arrive, hours=i)
    weather_hours.append(forecast.get_weather_at(forecastDate))

print(weather_hours)

time = 0
for weather in weather_hours:
    description = weather.get_detailed_status()
    clouds = weather.get_clouds()
    temperature = weather.get_temperature()['temp'] * 9/5 - 459.67
    wind = weather.get_wind()['speed']

    print('UTC time   ', '\t', time)
    print('description: ', '\t', description)
    print('cloud cover: ', '\t', clouds)
    print('temperature: ', '\t', temperature)
    print('wind speed: ', '\t', wind)
    print('\n')
    time = time + 3
try:
    rain = weather.get_rain()['all']
except KeyError:
    rain = 0

# OTHER MEASUREMENTS
# weather.get_dewpoint
# weather.get_heat_index
# weather.get_humidex
# weather.get_humidity
# weather.get_pressure
# weather.get_status
# weather.get_snow
# weather.get_sunrise_time
# weather.get_sunset_time
# weather.get_visibility_distance


print('description: ', '\t', description)
print('cloud cover: ', '\t', clouds)
print('temperature: ', '\t', temperature)
print('wind: ', '\t', wind)
print('rain: ', '\t', rain)

