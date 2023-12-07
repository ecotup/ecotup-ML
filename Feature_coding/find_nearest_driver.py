# backend_service.py (Flask app)

# %pip install --upgrade sqlalchemy pymysql

from flask import Flask, request, jsonify
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text


app = Flask(__name__)

# Set up database connection (replace with your MySQL database information)
engine = create_engine('mysql+pymysql://Ecotup_Access:ecotup*@34.101.70.239/db_ecotup')

user_id = 6
def find_nearest_driver(user_id):
    # Query to get the user's location
    user_query = text('SELECT user_longitude, user_latitude FROM tbl_user WHERE user_id = :user_id').bindparams(user_id=user_id)
    with engine.connect() as connection:
        user_result = connection.execute(user_query)
        user_location = user_result.fetchone()
        print(user_location)

    # Assuming the user_location contains both user_longitude and user_latitude
    user_longitude, user_latitude = user_location

    # Query to find the nearest driver
    driver_query = text('''
        SELECT driver_id, driver_longitude, driver_latitude,
           6371 * acos(cos(radians(:user_latitude)) * cos(radians(driver_latitude)) *
           cos(radians(driver_longitude) - radians(:user_longitude)) + sin(radians(:user_latitude)) *
           sin(radians(driver_latitude))) AS distance
        FROM tbl_driver
        ORDER BY distance
        LIMIT 1
    ''').bindparams(user_longitude=user_longitude,user_latitude=user_latitude)

    # Execute the driver query
    with engine.connect() as connection:
        driver_result = connection.execute(driver_query)
        nearest_driver = driver_result.fetchone()

    if nearest_driver:
        print(nearest_driver)
        return {
            'user_id': user_id,
            'nearest_driver_id': nearest_driver[0],
            'driver_longitude': nearest_driver[1],
            'driver_latitude': nearest_driver[2],
            'distance': nearest_driver[3]
            #index 0 is driver id
            #index 1 is driver longitude
            #index 2 is driver latitude
            #index 3 is distance
        }
    else:
        return {'error': 'No drivers found'}

result = find_nearest_driver(user_id)
print(result)

@app.route('/find_nearest_driver', methods=['POST'])
def find_nearest_driver_endpoint():
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        if user_id is not None:
            result = find_nearest_driver(user_id)
            #result = {'result': 'some_result'}
            return jsonify(result)
        else:
            return jsonify({'error': 'User ID not provided'})

    except Exception as e:
        return jsonify({'error': str(e)})

#running the app
#if __name__ == '__main__':
    #app.run(host='127.0.0.1', port=3307)