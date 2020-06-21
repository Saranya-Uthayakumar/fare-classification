import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import timedelta
from datetime import datetime
from geopy.geocoders import Nominatim

def calculate_distance(filePath):
	trainResult=pd.read_csv(filePath)
	pick_lattitude=trainResult['pick_lat']
	pick_longitude=trainResult['pick_lon']
	drop_lattitude=trainResult['drop_lat']
	drop_longitude=trainResult['drop_lon']
	distanceArray=[]
	R = 6373.0
	for i in range(len(pick_lattitude)):
		dlat=math.radians(pick_lattitude[i])-math.radians(drop_lattitude[i])
		dlon=math.radians(pick_longitude[i])-math.radians(drop_longitude[i])
		a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(pick_lattitude[i])) * math.cos(math.radians(drop_lattitude[i])) * math.sin(dlon / 2) ** 2
		c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
		distance = R * c
		distanceArray.append(distance)

	dict1={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'], 'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'], 'distance':distanceArray,'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'],'fare':trainResult['fare'],'label':trainResult['label']}
	# dict1={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'], 'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'], 'distance':distanceArray,'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'],'fare':trainResult['fare']}
	df=pd.DataFrame(dict1)
	df.to_csv('train_distance.csv')
	# df.to_csv('test_distance.csv')


def extract_feature_from_date_time(filePath):
	timeSpentArray=[]
	pickupDayArray=[]
	dropDayArray=[]
	travelDayArray=[]
	YMD='%m/%d/%Y'
	hms='%H:%M'
	travelHour=[]
	trainResult=pd.read_csv(filePath)
	pickupTime=trainResult['pickup_time']
	drop_time=trainResult['drop_time']
	# for i in range(len(pickupTime.values)):
	#     # print(drop_time[i].split(' ')[0])
	#     day=datetime.strptime(pickupTime[i].split(' ')[0],YMD).strftime('%w')
	#     time = datetime.strptime(drop_time[i].split(' ')[1], hms).strftime('%H')
	#     travelHour.append(time)
	#     # print(time)
	#     pickupDayArray.append(int(day))
	#     # print(day)

	for i in range(len(drop_time.values)):
	    # print(drop_time[i].split(' ')[0])
	    day=datetime.strptime(drop_time[i].split(' ')[0],YMD).strftime('%w')
	    # dropDayArray.append(int(day))
	    travelDayArray.append(int(day))

	timeSpentOnJourney=calculate_time_spent_on_journey(filePath)
	# dict2={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'],'timeSpentOnJourney':timeSpentOnJourney,'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'],'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'], 'pickupDay':pickupDayArray,'distance':trainResult['distance'],'travelHour':travelHour,'dropDay':dropDayArray,'fare':trainResult['fare'], 'label':trainResult['label']}
	dict2={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'],'timeSpentOnJourney':timeSpentOnJourney,'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'],'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'],'distance':trainResult['distance'],'travelDay':travelDayArray,'fare':trainResult['fare'], 'label':trainResult['label']}
	# dict2={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'],'timeSpentOnJourney':timeSpentOnJourney,'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'],'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'], 'pickupDay':pickupDayArray,'distance':trainResult['distance'],'travelHour':travelHour,'dropDay':dropDayArray,'fare':trainResult['fare']}
	# dict2={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'],'timeSpentOnJourney':timeSpentOnJourney,'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'],'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'],'distance':trainResult['distance'],'travelDay':travelDayArray,'fare':trainResult['fare']}
	df=pd.DataFrame(dict2)
	# df.to_csv('train_distance_times_days_lat_long.csv')
	df.to_csv('train_distance_time_day_lat_long.csv')
	# df.to_csv('test_distance_times_days_lat_long.csv')
	# df.to_csv('test_distance_time_day_lat_long.csv')

	# dict1={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'],'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'],'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'], 'pickupDay':pickupDayArray,'distance':trainResult['distance'],'travelHour':travelHour,'dropDay':dropDayArray,'fare':trainResult['fare'], 'label':trainResult['label']}
	# df=pd.DataFrame(dict1)
	# df.to_csv('train_distance_time_day_lat_long.csv')

def calculate_duration(filePath):
	FMT = '%H:%M'
	trainResult=pd.read_csv(filePath)
	pickupTime=trainResult['pickup_time']
	drop_time=trainResult['drop_time']
	duration=trainResult['duration']
	for i in range(len(duration.values)):
	    if(math.isnan(duration.values[i])):
	        tdelta=datetime.strptime(drop_time[i].split(' ')[1], FMT) - datetime.strptime(pickupTime[i].split(' ')[1], FMT)
	        if tdelta.days < 0:
	            tdelta = timedelta(days=0,
	                               seconds=tdelta.seconds, microseconds=tdelta.microseconds)
	        duration[i]=tdelta.seconds

	dict1={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'], 'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'], 'distance':trainResult['distance'],'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'],'fare':trainResult['fare'],'label':trainResult['label']}
	# dict1={'tripid': trainResult['tripid'], 'additional_fare':trainResult['additional_fare'], 'duration':trainResult['duration'], 'meter_waiting':trainResult['meter_waiting'], 'meter_waiting_fare':trainResult['meter_waiting_fare'], 'meter_waiting_till_pickup':trainResult['meter_waiting_till_pickup'],'pickup_time':trainResult['pickup_time'],'drop_time':trainResult['drop_time'], 'distance':trainResult['distance'],'pick_lat':trainResult['pick_lat'],'pick_lon':trainResult['pick_lon'], 'drop_lat':trainResult['drop_lat'],'drop_lon':trainResult['drop_lon'],'fare':trainResult['fare']}
	df=pd.DataFrame(dict1)
	df.to_csv('train_distance_duration.csv')
	# df.to_csv('test_distance_duration.csv')

def calculate_time_spent_on_journey(filePath):
	trainResult=pd.read_csv(filePath)
	duration=trainResult['duration']
	meterWaiting=trainResult['meter_waiting']
	timeSpentArray=[]
	for i in range(len(meterWaiting.values)):
	    if math.isnan(duration.values[i]):
	        timeSpentArray.append(duration.values[i])
	    else:
	        timeSpentOnJourney=duration[i]-meterWaiting[i]
	        timeSpentArray.append(timeSpentOnJourney)
	return timeSpentArray

def add_postal_code(filePath):
	trainResult=pd.read_csv(filePath)
	pick_lattitude=trainResult['pick_lat']
	pick_longitude=trainResult['pick_lon']
	drop_lattitude=trainResult['drop_lat']
	drop_longitude=trainResult['drop_lon']
	pickupLocation=[]
	dropLocation = []
	geolocator = Nominatim(user_agent="pickup" ,timeout=None)
	# geolocator1 = Nominatim(user_agent="drop")
	for i in range(len(pick_lattitude.values)):
	    location = geolocator.reverse(str(pick_lattitude[i])+', '+str(pick_longitude[i]))
	    pickupLocation.append(str(location.address.split(',')[-2]))
	    # time.sleep(1)
	    location1= geolocator.reverse(str(drop_lattitude[i]) + ', ' + str(drop_longitude[i]))
	    dropLocation.append(str(location1.address.split(',')[-2]))
	    # time.sleep(1)
	    dict1 = {'pickupLocation': pickupLocation, 'dropLocation':dropLocation}
	    df1 = pd.DataFrame(dict1)
	    df1.to_csv('pickup_drop_locations.csv')

def fill_missing_values(featureInput):
	imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	featureInputM=(imp.fit_transform(featureInput.values.reshape(-1,1))).ravel()
	return featureInputM

def feature_scale(featureInput):
	scaler = MinMaxScaler(feature_range=(-1,1))
	return (scaler.fit_transform(featureInput.values.reshape(-1,1))).ravel()




