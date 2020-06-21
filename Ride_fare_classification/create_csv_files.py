import pandas as pd
import numpy as np
from utils import *


calculate_distance('train.csv')
# calculate_distance('test.csv')
calculate_duration('train_distance.csv')
# calculate_duration('test_distance.csv')
extract_feature_from_date_time('train_distance.csv')
# extract_feature_from_date_time('test_distance.csv')

result1=pd.read_csv('train_distance_time_day_lat_long.csv')
# result5=pd.read_csv('test_distance_time_day_lat_long.csv')

additional_fareM=fill_missing_values(result1['additional_fare'])
durationM=fill_missing_values(result1['duration'])
meter_waitingM=fill_missing_values(result1['meter_waiting'])
meter_waiting_fareM=fill_missing_values(result1['meter_waiting_fare'])
meter_waiting_till_pickupM=fill_missing_values(result1['meter_waiting_till_pickup'])
timeSpentOnJourneyM=fill_missing_values(result1['timeSpentOnJourney'])
fareM=fill_missing_values(result1['fare'])

dict2={'tripid': result1['tripid'], 'additional_fare':additional_fareM, 'duration':durationM,'timeSpentOnJourney':timeSpentOnJourneyM,'meter_waiting':meter_waitingM, 'meter_waiting_fare':meter_waiting_fareM, 'meter_waiting_till_pickup':meter_waiting_till_pickupM,'pickup_time':result1['pickup_time'],'drop_time':result1['drop_time'],'pick_lat':result1['pick_lat'],'pick_lon':result1['pick_lon'], 'drop_lat':result1['drop_lat'],'drop_lon':result1['drop_lon'], 'pickupDay':result1['pickupDay'],'distance':result1['distance'],'travelHour':result1['travelHour'],'dropDay':result1['dropDay'],'fare':fareM, 'label':result1['label']}
# dict2={'tripid': result1['tripid'], 'additional_fare':additional_fareM, 'duration':durationM,'timeSpentOnJourney':timeSpentOnJourneyM,'meter_waiting':meter_waitingM, 'meter_waiting_fare':meter_waiting_fareM, 'meter_waiting_till_pickup':meter_waiting_till_pickupM,'pickup_time':result1['pickup_time'],'drop_time':result1['drop_time'],'pick_lat':result1['pick_lat'],'pick_lon':result1['pick_lon'], 'drop_lat':result1['drop_lat'],'drop_lon':result1['drop_lon'], 'travelDay':result1['travelDay'],'distance':result1['distance'],'fare':fareM, 'label':result1['label']}
df=pd.DataFrame(dict2)
df.to_csv('train_distance_time_day_lat_long_missing_mean.csv')
#
# dict1={'tripid': result5['tripid'], 'additional_fare':result5['additional_fare'], 'duration':result5['duration'],'meter_waiting':result5['meter_waiting'], 'meter_waiting_fare':result5['meter_waiting_fare'], 'meter_waiting_till_pickup':result5['meter_waiting_till_pickup'],'pickup_time':result5['pickup_time'],'drop_time':result5['drop_time'],'pick_lat':result5['pick_lat'],'pick_lon':result5['pick_lon'], 'drop_lat':result5['drop_lat'],'drop_lon':result5['drop_lon'], 'pickupDay':result5['pickupDay'],'distance':result5['distance'],'timeSpentOnTravel':result5['timeSpentOnJourney'],'travelHour':result5['travelHour'],'dropDay':result5['dropDay'],'fare':result5['fare']}
# dict1={'tripid': result5['tripid'], 'additional_fare':result5['additional_fare'], 'duration':result5['duration'],'meter_waiting':result5['meter_waiting'], 'meter_waiting_fare':result5['meter_waiting_fare'], 'meter_waiting_till_pickup':result5['meter_waiting_till_pickup'],'pickup_time':result5['pickup_time'],'drop_time':result5['drop_time'],'pick_lat':result5['pick_lat'],'pick_lon':result5['pick_lon'], 'drop_lat':result5['drop_lat'],'drop_lon':result5['drop_lon'], 'travelDay':result5['travelDay'],'distance':result5['distance'],'timeSpentOnTravel':result5['timeSpentOnJourney'],'fare':result5['fare']}
# df1=pd.DataFrame(dict1)
# df1.to_csv('test_distance_day_time_lat_long.csv')


# Postal code

# result2=pd.read_csv('train_distance_time_day_lat_long_postal_code.csv')
# additional_fareM=fill_missing_values(result1['additional_fare'])
# durationM=fill_missing_values(result1['duration'])
# meter_waitingM=fill_missing_values(result1['meter_waiting'])
# meter_waiting_fareM=fill_missing_values(result1['meter_waiting_fare'])
# meter_waiting_till_pickupM=fill_missing_values(result1['meter_waiting_till_pickup'])
# timeSpentOnJourneyM=fill_missing_values(result1['timeSpentOnJourney'])
# fareM=fill_missing_values(result1['fare'])
#
# dict2={'tripid': result2['tripid'], 'additional_fare':additional_fareM, 'duration':durationM,'timeSpentOnJourney':timeSpentOnJourneyM,'meter_waiting':meter_waitingM, 'meter_waiting_fare':meter_waiting_fareM, 'meter_waiting_till_pickup':meter_waiting_till_pickupM,'pickup_time':result2['pickup_time'],'drop_time':result2['drop_time'],'pick_lat':result2['pick_lat'],'pick_lon':result2['pick_lon'], 'drop_lat':result2['drop_lat'],'drop_lon':result2['drop_lon'], 'pickupDay':result12['pickupDay'],'distance':result1['distance'],'travelHour':result2['travelHour'],'dropDay':result2['dropDay'],'postalCode':result2['postalCode'],fare':fareM, 'label':result2['label']}
# # dict2={'tripid': result1['tripid'], 'additional_fare':additional_fareM, 'duration':durationM,'timeSpentOnJourney':timeSpentOnJourneyM,'meter_waiting':meter_waitingM, 'meter_waiting_fare':meter_waiting_fareM, 'meter_waiting_till_pickup':meter_waiting_till_pickupM,'pickup_time':result1['pickup_time'],'drop_time':result1['drop_time'],'pick_lat':result1['pick_lat'],'pick_lon':result1['pick_lon'], 'drop_lat':result1['drop_lat'],'drop_lon':result1['drop_lon'], 'travelDay':result1['travelDay'],'distance':result1['distance'],'fare':fareM, 'label':result1['label']}
# df=pd.DataFrame(dict2)
# df.to_csv('train_distance_time_day_lat_long_postal_code_missing_mean.csv')
