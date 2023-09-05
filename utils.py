import json, os, re, scipy
import pandas as pd
import numpy as np
import geopandas as gpd

from sklearn.metrics.pairwise import haversine_distances
from direction import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

TYPE_3_VOLTAGE = 400
EARTH_RADIUS_KM = 6371

def load_charging_stations():
    """Loads open charge map charging station data and stores a list of all charging stations with type 3 charging capabilities.

    Returns:
        list: list of dictionaries. Each dictionary contains information about 1 charging station. 
    """
    data = []
    for file in os.listdir('data/stations'):
        filename = os.fsdecode(file)
        if filename[-5:] == '.json':
            with open('data/stations/' + filename, 'r') as json_file:
                station = json.load(json_file)
                for connection in station:
                    try:
                        if connection['Voltage'] >= TYPE_3_VOLTAGE:
                            data.append(station)
                    finally:
                        continue

    return data

def load_emissions_timeseries():
    """Loads electricity maps emissions data from 2022

    Returns:
        dict: a string abbreviation zone maps to a DataFrame emissions timeseries for 2022
    """

    timeseries = {}
    for filename in os.listdir('data/timeseries'):
        if filename[-3:] == 'csv':
            timeseries[re.match(r'^[^_]+', filename).group()] = pd.read_csv('data/timeseries/' + filename)
    
    return timeseries

def load_directions():
    """Loads google maps directions

    Returns:
        list: a list of Direction objects
    """
    directions = []

    for filename in os.listdir('data/directions/points'):
        route = filename[:-4]
        points = pd.read_csv('data/directions/points/' + filename)
        steps = pd.read_csv('data/directions/steps/' + filename)

        directions.append(Direction(route, points, steps))

    return directions

def load_cars():
    """Loads data about cars

    Returns:
        DataFrame: DataFrame where each row contains performance information about 1 EV
    """
    return pd.read_csv('data/cars.csv')


def load_zone_geometries():
    """Loads geometries of zones

    Returns:
        GeoDataFrame: geopandas DataFrame containing columns zoneName, countryKey, countryName, geometry
    """
    gdf = gpd.read_file('data/world.geojson')
    gdf = gdf[gdf['countryName'] == 'United States']
    return gdf


def interpolate(interval, trip):
    """Interpolates driving route.

    Args:
        interval (float): distance (in km) desired between interpolated points
        trip (DataFrame): DataFrame of waypoints with a "lon" column and a "lat" column

    Returns:
        DataFrame: the interpolated path stored in a DataFrame with 2 columns: "lat" and "lon"
    """
    points = np.deg2rad(np.array(trip))

    dists = [0]
    for i in range(len(points) - 1):
        dists.append(haversine_distances(points[i, :].reshape(1, -1), points[i+1, :].reshape(1, -1))[0, 0] * 6371)
                    
    trip['dist'] = dists
    trip['cumdist'] = trip['dist'].cumsum()
    
    dold = list(trip["cumdist"])
    xold = list(trip["lon"])
    yold = list(trip["lat"])
    
    f1 = scipy.interpolate.interp1d(dold,xold,kind = 'linear')
    f2 = scipy.interpolate.interp1d(dold,yold,kind = 'linear')
    dnew = np.linspace(dold[0], dold[-1], num = round(dold[-1] / interval), endpoint = True)
        
    xnew = list(f1(dnew))
    ynew = list(f2(dnew))

    return pd.DataFrame({'lat' : ynew, 'lon' : xnew})

def filter_charging_stations(stations, points):
    """Filters charging stations to only cointain those withing the bounding box of the route

    Args:
        stations (list): list of dictionaries. Each dictionary contains information about 1 charging station. 
        points (DataFrame): DataFrame containing 'lat' and 'lon' columns corresoponding to waypoints 

    Returns:
        list: list of charging stations within the bounding box
    """
    min_lat = min(points['lat'])
    max_lat = max(points['lat'])
    min_lon = min(points['lon'])
    max_lon = max(points['lon'])
    stations_in_bbx = []

    for station in stations:
        if min_lat <= station['AddressInfo']['Latitude'] <= max_lat and min_lon <= station['AddressInfo']['Longitude'] <= max_lon:
            stations_in_bbx.append(station)

    return stations_in_bbx

def get_zones(df, gdf):
    """Gets zones visited at each (lat, lon) timestep in df with geographic information from gdf

    Args:
        df (DataFrame):  DataFrame containing 'lat' and 'lon' columns corresoponding to waypoints 
        gdf (GeoDataFrame): GeoDataFrame containing 'zoneName' and their corresponding 'geometry'

    Returns:
        list: list of zones visited in order
    """
    zones = []
    for lat, lon in zip(df['lat'], df['lon']):
        point = Point(lon, lat)
        zones.append(None)
        for zone, geo in zip(gdf['zoneName'], gdf['geometry']):
            if geo.contains(point):
                zones[-1] = zone
                break

    # Get zone from nearest timestep if point does not fall in a zone
    missing_zone = np.where(np.array(zones) == None)[0]
    for i in missing_zone:
        diff = 1
        while zones[i] == None and i - diff >= 0 and i + diff < len(zones):
            if zones[i - diff]:
                zones[i] = zones[i - diff]
                break
            elif zones[i + diff]:
                zones[i] = zones[i + diff]
                break
            
            diff += 1

    return zones

def get_timestamps(points, steps):
    """Returns time series of time elapsed (seconds) since start of trip

    Args:
        points (DataFrame): DataFrame containing 'lat' and 'lon' columns corresoponding to waypoints 
        steps (DataFrame): DataFrame containing information about legs of the trip and their durations. Each leg corresponds to driving on a different road

    Returns:
        numpy array: time series of time elapsed (seconds) since start of trip
    """

    timestamps = np.repeat(-1, len(points))
    timestamps[0] = 0

    lat_series = np.deg2rad(points['lat'])
    lon_series = np.deg2rad(points['lon'])

    for i in range(len(steps)):    
        end_loc = np.array([np.deg2rad(steps['end_location.lat'][i]), np.deg2rad(steps['end_location.lng'][i])]).reshape(1, -1)
            
        dists = haversine_distances(np.array([lat_series, lon_series]).T, end_loc) * EARTH_RADIUS_KM 
        argmin_dist = np.argmin(dists)
        
        timestamps[argmin_dist] += steps['duration.value'][i]

    # Linear interpolation of speed in between legs in directions
    waypoints = np.where(timestamps != -1)[0]
    prev_w = waypoints[0]

    for w in waypoints[1:]:
        timestamps[prev_w + 1 : w + 1] = np.repeat(timestamps[w] / (w - prev_w), w - prev_w)
        prev_w = w

    timestamps = np.cumsum(timestamps)
    timestamps = timestamps - timestamps[0]

    return timestamps