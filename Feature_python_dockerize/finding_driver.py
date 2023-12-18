# -*- coding: utf-8 -*-
"""Finding_Driver.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W1rF6HsDpTJlnU_2lR4U916Q3mDCrduD

# **THE APP**
"""


# %pip install --upgrade sqlalchemy pymysql

from flask import Flask, request, jsonify,Blueprint
import pymysql
from sqlalchemy import create_engine, text





def find_nearest_driver(user_id):

    engine = create_engine('mysql+pymysql://Ecotup_Access:ecotup*@34.101.70.239/db_ecotup')
    # Query to get the user's location
    user_query = text('SELECT user_longitude, user_latitude FROM tbl_user WHERE user_id = :user_id').bindparams(user_id=user_id)
    with engine.connect() as connection:
        user_result = connection.execute(user_query)
        user_location = user_result.fetchone()
        print(user_location)


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



finding_driver_bp = Blueprint('find_nearest_driver', __name__)
@finding_driver_bp.route('/find_nearest_driver/<int:user_id>', methods=['GET'])
def find_nearest_driver_endpoint(user_id):
    print('hello find_nearest_driver!')
    try:
        #data = request.get_json()
        #user_id = data.get('user_id')

        #if user_id is not None:
        result = find_nearest_driver(user_id)
        return jsonify(result)
        #else:
        #   return jsonify({'error': 'User ID not provided'})

    except Exception as e:
        return jsonify({'error': str(e)})



#running the app
#if __name__ == '__main__':
    #app.run(host='127.0.0.1', port=330