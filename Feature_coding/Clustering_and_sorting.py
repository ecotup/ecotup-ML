#download sqlalchemy and pymysql
#%pip install --upgrade sqlalchemy pymysql

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import tensorflow as tf
import zipfile
import os
import joblib
from math import radians, sin, cos, sqrt, atan2
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans


# **Take the user data from SQL**
def fetch_data_from_mysql():
    app = Flask(__name__)


    engine = create_engine('mysql+pymysql://Ecotup_Access:ecotup*@34.101.70.239/db_ecotup')

    get_data_query = '''SELECT
    u.user_longitude,
    u.user_latitude
FROM
    tbl_user u
JOIN
    tbl_subscription s ON u.subscription_id = s.subscription_id
WHERE
    s.subscription_value > 0
ORDER BY
    u.user_id ASC; '''

    # Fetch data into a Pandas DataFrame
    data_input = engine.connect().execute(text(get_data_query))
    result_data = data_input.fetchall()
    return result_data

#input_data_from_sql = fetch_data_from_mysql()

#print(input_data_from_sql)

# Generate random coordinates for 20 houses
#np.random.seed(42)
#houses_set1 = [(np.random.uniform(-7, -6), np.random.uniform(106, 107)) for _ in range(20)]
#houses_set2 = [(np.random.uniform(-10, -11), np.random.uniform(132, 133)) for _ in range(20)]
#houses_set3 = [(np.random.uniform(-15, -16), np.random.uniform(92, 93)) for _ in range(20)]


#Do Clustering after getting the data

#Assign Centroids
def kMeans_init_centroids(X, K):
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids

#Function to assign each data point to nearest centroid (Clustering)
def find_closest_centroids(X, centroids):

    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distance = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distance)

    return idx

#Update position of the centroid every iteration
def compute_centroids(X, idx, K):
    
    centroids = np.zeros((K, X.shape[1]))

    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0)

    return centroids

#Function For the training
def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):

        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, iteration):
            plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis', label='Data Points')
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centroids')
            plt.scatter(previous_centroids[:, 0], previous_centroids[:, 1], marker='o', s=100, color='blue', alpha=0.5, label='Previous Centroids')
            plt.title(f'K-Means Clustering - Iteration {iteration}')
            plt.xlabel('Latitude')
            plt.ylabel('Longitude')
            plt.legend()
            plt.show()
        # Optionally plot progress
        if plot_progress:

            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show()
    return centroids, idx

#Training
def Clustering_Training(result_data) :
    X = np.array(result_data)
    K = 3
    max_iters = 20
# Set initial centroids by picking random examples from the dataset
    initial_centroids = kMeans_init_centroids(X, K)
    X_with_centroids = np.vstack((X, initial_centroids))
# Append centroids to the dataset

    #print(X_with_centroids)
# Run K-Means
    centroids, idx = run_kMeans(X_with_centroids, initial_centroids, max_iters, plot_progress=True)
    #print("this is the centroids: ",centroids)
    #print("this is the cluster assignment: ",idx)
    clustered_data = np.column_stack((X_with_centroids, idx))
    #print(clustered_data)
    #print(len(clustered_data))

    return clustered_data
#result_of_the_cluster = Clustering_Training()



#Update the database
def update_sql_with_clusters(clustered_data):
  try:
    # ... your code ...
    engine = create_engine('mysql+pymysql://Ecotup_Access:ecotup*@34.101.70.239/db_ecotup')
    with engine.connect() as connection:
        take_user_id = connection.execute(text("SELECT user_id FROM tbl_user ORDER BY user_id ASC"))
        user_ids = [row.user_id for row in take_user_id]

        result = connection.execute(text("SELECT MAX(cluster_id) FROM tbl_cluster"))
        max_cluster_id = result.scalar()


        # If there are no existing cluster_ids, set max_cluster_id to 0
        if max_cluster_id is None:
            max_cluster_id = 0

        # update cluster_id to NULL in tbl_user
        update_user_query = text("UPDATE tbl_user SET cluster_id = NULL")
        connection.execute(update_user_query)

        # delete existing rows in tbl_cluster
        delete_cluster_query = text("DELETE FROM tbl_cluster")
        connection.execute(delete_cluster_query)



        for user_id, row in zip(user_ids, clustered_data):
            user_longitude, user_latitude, cluster_assignment = row[:3]  # Assuming the first three columns are longitude, latitude, and cluster_assignment


            #to make it start from 1

            # Construct the new cluster name
            cluster_name = f"cluster {cluster_assignment}"
            # Insert the data first into tbl_cluster table
            cluster_query = text("""
                INSERT INTO tbl_cluster ( cluster_name, cluster_region,  user_id, driver_id)
                VALUES ( :cluster_name, :cluster_name, :user_id,
                CASE
                    WHEN :cluster_name = 'cluster 0.0' THEN 2
                    WHEN :cluster_name = 'cluster 1.0' THEN 3
                    WHEN :cluster_name = 'cluster 2.0' THEN 4


                    END
                )
                """)
            connection.execute(cluster_query, { 'cluster_name': cluster_name,
                                               'user_id': user_id
                                               })

           # Update tbl_user using a JOIN query
            user_cluster_query = text("""
                UPDATE tbl_user
                JOIN tbl_cluster ON tbl_user.user_id = tbl_cluster.user_id
                SET tbl_user.cluster_id = tbl_cluster.cluster_id
                WHERE tbl_user.user_id = :user_id
                """)
            connection.execute(user_cluster_query,{'user_id': user_id})
            connection.commit()

  except Exception as e:
    print(f"Error: {e}")
    # Get the maximum existing cluster_id in the tbl_cluster table


#Take the drivers data
def get_drivers_data():
    query = '''
SELECT
    d.driver_id,
    d.driver_longitude,
    d.driver_latitude

FROM
    tbl_driver d;
'''

    driver_Data = engine.connect().execute(text(query)).fetchall()

    driver_data_array = np.array(driver_Data)


    selected_columns = driver_data_array[:, 1:4]

    # Indices of rows i want to keep
    indices_to_keep = [0, 1, 2]

    # Extract only the specified rows
    driver_selected_rows = driver_data_array[indices_to_keep]


    #use this
    driver1 =driver_selected_rows[0]
    driver2 =driver_selected_rows[1]
    driver3 =driver_selected_rows[2]

    driver_selected_rows_noID = driver_selected_rows[:, 1:]
    #print(driver_selected_rows_noID)

    return driver1,driver2,driver3

#Take the clusters into 3 variables
def divide_clusters(clustered_data):

    # Add user_id at the front of each row
    result_of_the_cluster_with_id = np.c_[np.arange(1, len(clustered_data) + 1), clustered_data]

    # Get unique cluster assignments
    unique_clusters = np.unique(result_of_the_cluster_with_id[:, 3])

    # Create a dictionary to store data for each cluster
    clustered_data_dict = {cluster: result_of_the_cluster_with_id[result_of_the_cluster_with_id[:, 3] == cluster] for cluster in unique_clusters}

    # Create variables for each cluster
    cluster0 = clustered_data_dict[unique_clusters[0]]
    cluster1 = clustered_data_dict[unique_clusters[1]]
    cluster2 = clustered_data_dict[unique_clusters[2]]

    cluster0_withoutid = cluster0[:, :2]
    cluster1_withoutid = cluster1[:, :2]
    cluster2_withoutid = cluster2[:, :2]

    return cluster0,cluster1,cluster2

#Calculate and sort
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance
def calculate_distance_matrix(points):
    """
    Calculate distance matrix between points using haversine formula.
    """
    num_points = len(points)
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            distance_matrix[i, j] = haversine(points[i][0], points[i][1], points[j][0], points[j][1])

    return distance_matrix

def tsp_greedy(distance_matrix):
    """
    Solve the Traveling Salesman Problem using a greedy algorithm.
    """
    num_points = len(distance_matrix)
    unvisited_points = set(range(1, num_points))  # Exclude the starting point
    visited_points = [0]  # Starting from the first point

    while unvisited_points:
        current_point = visited_points[-1]
        nearest_point = min(unvisited_points, key=lambda x: distance_matrix[current_point, x])
        visited_points.append(nearest_point)
        unvisited_points.remove(nearest_point)

    return visited_points

def assign_and_sort_with_tsp_greedy(cluster_data, vehicle_data):

    # Manually assign the vehicle to the cluster
    vehicle_id = vehicle_data[0]
    cluster_id = cluster_data[0, 0]

    # Find nearest home point in the cluster
    distances_to_vehicle = [haversine(vehicle_data[1], vehicle_data[2], point[1], point[2]) for point in cluster_data]
    nearest_home_point_index = np.argmin(distances_to_vehicle)

    #print("Distances to Vehicle:", distances_to_vehicle)
    #print("Nearest Home Point Index:", nearest_home_point_index)

    if nearest_home_point_index < len(cluster_data):
        nearest_home_point = cluster_data[nearest_home_point_index, 1:]

        # Pre-calculate distances between all points in the cluster
        cluster_points = cluster_data[:, 1:]
        distance_matrix = calculate_distance_matrix(cluster_points)

        # Sort points based on distances in ascending order
        sorted_indices_ascending = np.argsort(distances_to_vehicle)
        sorted_distance_matrix_ascending = distance_matrix[sorted_indices_ascending][:, sorted_indices_ascending]

        # Use the greedy TSP algorithm to find the optimal path
        tsp_path_indices = tsp_greedy(sorted_distance_matrix_ascending)

        # Collect results in an array
        result_array = np.array([[
            tsp_path_index,
            vehicle_id, vehicle_data[1], vehicle_data[2],
            cluster_data[sorted_indices_ascending[tsp_path_index], 0],
            cluster_data[sorted_indices_ascending[tsp_path_index], 1],
            cluster_data[sorted_indices_ascending[tsp_path_index], 2]
        ] for tsp_path_index in tsp_path_indices])


        # Display results
        #print(f"Assigned Vehicle {vehicle_id} to Cluster {cluster_id}")
        #print(vehicle_data)
        #print(f"Nearest Home Point in Cluster {cluster_id}:", nearest_home_point)
        #print(f"Optimal TSP Path Indices in Cluster {cluster_id}:", tsp_path_indices)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(cluster_data[:, 1], cluster_data[:, 2], label="Cluster Points")



        plt.scatter(vehicle_data[1], vehicle_data[2], color='red', label=f"Vehicle {vehicle_id}")
        plt.scatter(nearest_home_point[0], nearest_home_point[1], color='green', label="Nearest Home Point")

        # Labeling TSP Path points with order
        tsp_path_coordinates = cluster_data[sorted_indices_ascending[tsp_path_indices], 1:]
        for i, point_index in enumerate(tsp_path_indices):
            plt.text(tsp_path_coordinates[i][0], tsp_path_coordinates[i][1], str(point_index), fontsize=8, color='black', ha='right', va='bottom')

        plt.title(f"Vehicle {vehicle_id} Assignment in Cluster {cluster_id}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()

        return result_array
    else:
        print("Error: Nearest Home Point Index out of bounds.")


#to send the result
clustering_sorting = Flask(__name__)

@clustering_sorting.route('/clustering_and_sorting', methods=['GET'])
def clustering_and_sorting_endpoint():
    try:
        engine = create_engine('mysql+pymysql://Ecotup_Access:ecotup*@34.101.70.239/db_ecotup')

        # Fetch data from MySQL and assign it to result_data
        result_data = fetch_data_from_mysql()

        # Perform clustering
        clustered_data = Clustering_Training(result_data)

        # Update SQL with clustered data
        update_sql_with_clusters(clustered_data)

        # Get drivers data
        driver1,driver2,driver3 = get_drivers_data()

        # Divide clusters
        cluster0,cluster1,cluster2 = divide_clusters(clustered_data)

        # Assign and sort with TSP greedy algorithm for each cluster and driver
        result_array_cluster0 = assign_and_sort_with_tsp_greedy(cluster0, driver1)
        print(result_array_cluster0)
        result_array_cluster1 = assign_and_sort_with_tsp_greedy(cluster1, driver2)
        print(result_array_cluster1)
        result_array_cluster2 = assign_and_sort_with_tsp_greedy(cluster2, driver3)
        print(result_array_cluster2)
        return jsonify(result_array_cluster0,result_array_cluster1,result_array_cluster2)



    except Exception as e:
        return jsonify({'error': str(e)})