import utils
from direction import *
import optimizer

import datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

MAX_DEVIATION = 10
EARTH_RADIUS_KM = 6371

INTERPOLATION_DISTANCE = 0.1

CHADEMO = 2
TESLA_SUPERCHARGER = 27
TESLA = 30
CCS_TYPE_1 = 32
CCS_TYPE_2 = 33

charging_stations = utils.load_charging_stations()
emissions_timeseries = utils.load_emissions_timeseries()
directions = utils.load_directions()
cars = utils.load_cars()
gdf = utils.load_zone_geometries()

for d in directions:
    points = utils.interpolate(INTERPOLATION_DISTANCE, d.points)
    steps = d.steps

    current_chargers = utils.filter_charging_stations(points)

    # Getting waypoints where chargers are available 
    valid_chargers = []
    chademo_charger_available = np.zeros(len(points))
    ccs_charger_available = np.zeros(len(points))
    tesla_charger_available = np.zeros(len(points))


    lat_series = np.deg2rad(points['lat'])
    lon_series = np.deg2rad(points['lon'])

    for station in current_chargers:    

        station_loc = np.array([np.deg2rad(station['AddressInfo']['Latitude']), np.deg2rad(station['AddressInfo']['Longitude'])]).reshape(1, -1)
            
        dists = haversine_distances(np.array([lat_series, lon_series]).T, station_loc) * EARTH_RADIUS_KM
            
        min_dist = min(dists)
        argmin_dist = np.argmin(dists)

        timestamps = utils.get_timestamps(points, steps)
        points['timestamp'] = timestamps
        points['hour'] = df['timestamp'] // 3600

        
        if min_dist <= MAX_DEVIATION:
            connections = station['Connections']
            for c in connections:
                try:
                    charger_id = c['ConnectionTypeID']
                    if charger_id == CHADEMO:
                        chademo_charger_available[argmin_dist] = 1
                    elif charger_id == TESLA or charger_id == TESLA_SUPERCHARGER:
                        tesla_charger_available[argmin_dist] = 1
                    elif charger_id == CCS_TYPE_1 or charger_id == CCS_TYPE_2:
                        ccs_charger_available[argmin_dist] = 1
                except:
                    pass

        points['chademo_charger_available'] = chademo_charger_available
        points['ccs_charger_available'] = ccs_charger_available
        points['tesla_charger_available'] = tesla_charger_available
        
        zones = utils.get_zones(points, gdf)
        points['zone'] = zones

        emissions = []
        start_time = dt.datetime(2022, 7, 1, hour=6)
        for hour, zone in zip(points['hour'], points['zone']):
            zone_df = emissions_timeseries[zone]
            curr_time = start_time + dt.timedelta(hours=hour)
            curr_time = curr_time.strftime('%Y-%m-%d %H:%M:%S')
            emissions.append(zone_df[zone_df['Datetime (UTC)'] == curr_time]['Carbon Intensity gCO₂eq/kWh (direct)'].iloc[0])

        points['emissions'] = emissions
        points['cumdist'] = np.arange(0, len(points) * INTERPOLATION_DISTANCE, INTERPOLATION_DISTANCE)

        chademo_only = points[points['chademo_charger_available'] == 1]
        ccs_only = points[points['ccs_charger_available'] == 1]
        tesla_only = points[points['tesla_charger_available'] == 1]

        results = pd.DataFrame(columns=["Model", "Emissions"])

        for _, row in cars:
            model = row["Model"]
            capacity = row['Capacity (kWh)']
            kwh_per_km = row['kWh / km']
            charger_type = row['Charger']

            if charger_type == 'CHADEMO':
                charging_timeseries = optimizer.optimize(chademo_only, capacity, kwh_per_km)
            elif charger_type == 'CCS':
                charging_timeseries = optimizer.optimize(ccs_only, capacity, kwh_per_km)
            else:
                charging_timeseries = optimizer.optimize(tesla_only, capacity, kwh_per_km) 

            results.loc[len(results)] = [model, sum(results)]

            
