#### WORKING ####
import argparse
import joblib
import pandas as pd
import numpy as np

from prophet import Prophet

class Model:
    def __init__(self, model_path='trained_model.pkl'):
        """
        Initialize the model and load the pre-trained Prophet model from the specified file path.
        """
        self.model = joblib.load(model_path)
        print("Model loaded successfully from", model_path)

    def predict(self, reference_data, cleaned_data, results_df, threshold_multiplier=1.5):
        """
        Predict anomalies for each location using the trained model.

        Args:
            reference_data (pd.DataFrame): DataFrame containing reference locations.
            cleaned_data (pd.DataFrame): DataFrame containing the cleaned time-series data.
            results_df (pd.DataFrame): DataFrame containing mapping of reference locations to cleaned data points.
            threshold_multiplier (float): Multiplier for residual threshold to detect anomalies.

        Returns:
            pd.DataFrame: DataFrame containing formatted anomalies for all locations.
        """
        formatted_results = []
        aggregated_anomalies = []

        for location in reference_data['location'].unique():
            print(f"Processing predictions for location: {location}")

            location_pairs = results_df[results_df['Reference Location'] == location]
            location_anomalies = []

            for _, pair in location_pairs.iterrows():
                lat = pair['Cleaned Data Latitude']
                lon = pair['Cleaned Data Longitude']

                pair_data = cleaned_data[
                    (cleaned_data['latitude'] == lat) & (cleaned_data['longitude'] == lon)
                ]

                if pair_data.empty:
                    print(f"No data for Lat: {lat}, Lon: {lon}. Skipping...")
                    continue

                pair_data = pair_data.rename(columns={'ds': 'time', 'y': 'sla'})
                pair_data['time'] = pd.to_datetime(pair_data['time'], errors='coerce')
                pair_data = pair_data.dropna(subset=['sla', 'time'])
                pair_data = pair_data.rename(columns={'time': 'ds', 'sla': 'y'})

                try:
                    forecast = self.model.predict(pair_data[['ds', 'latitude', 'longitude']])
                    residuals = pair_data['y'].values - forecast['yhat'].values

                    threshold = threshold_multiplier * residuals.std()
                    anomalies = (abs(residuals) > threshold).astype(int)

                    anomaly_data = pd.DataFrame({
                        'date': pair_data['ds'],
                        'anomaly': anomalies
                    })
                    location_anomalies.append(anomaly_data)

                except Exception as e:
                    print(f"Error predicting anomalies for Lat: {lat}, Lon: {lon} -> {e}")
                    continue

            if location_anomalies:
                location_anomalies_df = pd.concat(location_anomalies).groupby('date')['anomaly'].sum().reset_index()
                location_anomalies_df['anomaly'] = (location_anomalies_df['anomaly'] >= 2).astype(int)
                location_anomalies_df.rename(columns={'anomaly': location}, inplace=True)
                aggregated_anomalies.append(location_anomalies_df)

        # Create a DataFrame for final results
        final_results = pd.DataFrame({'date': pd.date_range(start='1993-01-01', end='2013-12-31')})
        final_results['date'] = pd.to_datetime(final_results['date'])  # Ensure datetime format

        # Merge anomalies from each location into final_results
        for location_result in aggregated_anomalies:
            location_result['date'] = pd.to_datetime(location_result['date'])  # Ensure datetime format
            final_results = pd.merge(final_results, location_result, on='date', how='left')

        # Fill missing values with 0 for all locations
        final_results = final_results.fillna(0).astype(int)
        final_results['date'] = pd.to_datetime(final_results['date']).dt.strftime('%Y-%m-%d')

        return final_results


        
        return prediction_data
# Function to calculate Haversine distance using NumPy for vectorized operations
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def find_top3_nearest(reference_data, cleaned_data):
    """
    Finds the top 3 nearest unique points in cleaned_data for each location in reference_data.

    Args:
        reference_data (pd.DataFrame): DataFrame containing reference locations with 'latitude', 'longitude', and 'location'.
        cleaned_data (pd.DataFrame): DataFrame containing cleaned locations with 'latitude' and 'longitude'.

    Returns:
        pd.DataFrame: DataFrame containing the top 3 nearest points for each reference location.
    """
    # Precompute latitudes and longitudes from cleaned_data for faster access
    cleaned_latitudes = cleaned_data['latitude'].to_numpy()
    cleaned_longitudes = cleaned_data['longitude'].to_numpy()

    # DataFrame to store results
    results = []

    # Loop through each location in reference_data
    for idx, ref_row in reference_data.iterrows():
        print(f"Processing location {ref_row['location']} ({idx + 1}/{len(reference_data)})...")
        
        ref_lat = ref_row['latitude']
        ref_lon = ref_row['longitude']
        ref_location = ref_row['location']
        
        # Compute distances to all points in cleaned_data using vectorized operations
        distances = haversine_vectorized(ref_lat, ref_lon, cleaned_latitudes, cleaned_longitudes)
        
        # Create a DataFrame to store distances and coordinates for filtering
        distance_df = pd.DataFrame({
            'latitude': cleaned_latitudes,
            'longitude': cleaned_longitudes,
            'distance': distances
        })
        
        # Drop duplicate latitude/longitude pairs and sort by distance
        distance_df = distance_df.drop_duplicates(subset=['latitude', 'longitude']).sort_values(by='distance')
        
        # Select the top 3 closest unique points
        top3_unique = distance_df.head(3)
        
        # Append the results
        for i, row in enumerate(top3_unique.itertuples(), start=1):
            results.append({
                'Reference Location': ref_location,
                'Reference Latitude': ref_lat,
                'Reference Longitude': ref_lon,
                'Rank': i,
                'Cleaned Data Latitude': row.latitude,
                'Cleaned Data Longitude': row.longitude,
                'Distance (km)': row.distance
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

#Prepare Combined Data
def prepare_combined_data(reference_data, cleaned_data, results_df):
    combined_data = []

    for location in reference_data['location'].unique():
        print(f"Processing location: {location}")
        location_pairs = results_df[results_df['Reference Location'] == location]

        for _, pair in location_pairs.iterrows():
            lat = pair['Cleaned Data Latitude']
            lon = pair['Cleaned Data Longitude']

            # Filter data for the current lat/long pair
            pair_data = cleaned_data[
                (cleaned_data['latitude'] == lat) & (cleaned_data['longitude'] == lon)
            ]

            if pair_data.empty:
                continue

            # Add location-specific information to the dataset
            pair_data = pair_data.rename(columns={'ds': 'time', 'y': 'sla'})
            pair_data['time'] = pd.to_datetime(pair_data['time'], errors='coerce')
            pair_data = pair_data.dropna(subset=['sla', 'time'])
            pair_data['location'] = location
            pair_data['latitude'] = lat
            pair_data['longitude'] = lon

            combined_data.append(pair_data)

    # Combine all data into a single DataFrame
    combined_df = pd.concat(combined_data)
    combined_df = combined_df.rename(columns={'time': 'ds', 'sla': 'y'})
    return combined_df


def main(input_file, model_file, output_file=None):
    """
    Main function to process predictions and save results.

    Args:
        input_file (str): Path to the input data file.
        model_file (str): Path to the trained model file.
        output_file (str): Path to save the output predictions. Defaults to 'demo_sla.csv'.
    """
    # Default output file name
    if output_file is None:
        output_file = "demo_sla.csv"

    # Locations dataset
    data = [
        {"latitude": 43.658060, "longitude": -70.244170, "location": "Portland"},
        {"latitude": 40.700556, "longitude": -74.014167, "location": "The Battery"},
        {"latitude": 36.946701, "longitude": -76.330002, "location": "Sewells Point"},
        {"latitude": 41.361401, "longitude": -72.089996, "location": "New London"},
        {"latitude": 39.266944, "longitude": -76.579444, "location": "Baltimore"},
        {"latitude": 41.504333, "longitude": -71.326139, "location": "Newport"},
        {"latitude": 38.873000, "longitude": -77.021700, "location": "Washington"},
        {"latitude": 38.782780, "longitude": -75.119164, "location": "Lewes"},
        {"latitude": 40.466944, "longitude": -74.009444, "location": "Sandy Hook"},
        {"latitude": 39.356667, "longitude": -74.418053, "location": "Atlantic City"},
        {"latitude": 44.904598, "longitude": -66.982903, "location": "Eastport"},
        {"latitude": 32.036700, "longitude": -80.901700, "location": "Fort Pulaski"},
    ]

    reference_data = pd.DataFrame(data)
    input_data = pd.read_csv(input_file)  # Read input file
    cleaned_data = input_data.dropna()

    results_df = find_top3_nearest(reference_data, cleaned_data)

    # Convert and sort data
    cleaned_data['time'] = pd.to_datetime(cleaned_data['time'], errors='coerce')
    cleaned_data = cleaned_data.rename(columns={'time': 'ds', 'sla': 'y'})
    cleaned_data = cleaned_data.sort_values(by='ds').reset_index(drop=True)

    # Initialize the model
    model = Model(model_path=model_file)
    anomaly_results = model.predict(reference_data, cleaned_data, results_df)
    anomaly_results.to_csv(output_file, index=False)
    print(f"Anomaly results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict anomalies for each location using a trained Prophet model")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file")
    parser.add_argument('model_file', type=str, help="Path to the trained model file (.pkl)")
    parser.add_argument('--output_file', type=str, help="Path to save the output predictions CSV file (default: demo_sla.csv)", default=None)

    args = parser.parse_args()
    main(args.input_file, args.model_file, args.output_file)
    
                


