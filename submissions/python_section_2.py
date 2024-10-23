import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a distance matrix based on cumulative distances between toll locations,
    ensuring symmetry and setting diagonal values to 0.
    
    Args:
    - df (pd.DataFrame): DataFrame containing columns ['id_start', 'id_end', 'distance'].
    
    Returns:
    - pd.DataFrame: Symmetric distance matrix with cumulative distances.
    """
    unique_ids = pd.Index(pd.concat([df['id_start'], df['id_end']]).unique())
    n = len(unique_ids)
    
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)
    
    np.fill_diagonal(distance_matrix.values, 0)
    
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix


def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Unrolls the distance matrix into a long-format DataFrame with columns 
    id_start, id_end, and distance, excluding rows where id_start == id_end.

    Args:
    - distance_matrix (pd.DataFrame): The distance matrix generated from calculate_distance_matrix.

    Returns:
    - pd.DataFrame: A DataFrame with columns id_start, id_end, and distance.
    """
    data = []
    
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    unrolled_df = pd.DataFrame(data)

    return unrolled_df



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    return df


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates toll rates based on vehicle types and adds new columns to the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing id_start, id_end, and distance columns.

    Returns:
    - pd.DataFrame: DataFrame with additional columns for toll rates based on vehicle types.
    """
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df


import datetime
def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-based toll rates based on vehicle types and adds time-related columns to the DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame containing id_start, id_end, distance and vehicle rate columns.

    Returns:
    - pd.DataFrame: DataFrame with additional columns for time intervals and adjusted toll rates.
    """
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
    discount_factors = {
        'weekdays': {
            'morning': 0.8,  
            'day': 1.2,     
            'evening': 0.8  
        },
        'weekends': 0.7  
    }

    df['start_day'] = pd.Series(weekdays * (len(df) // len(weekdays)) + weekdays[:len(df) % len(weekdays)])
    df['end_day'] = df['start_day']  
    df['start_time'] = pd.Series([datetime.time(0, 0, 0)] * len(df))  
    df['end_time'] = pd.Series([datetime.time(23, 59, 59)] * len(df))  

    def apply_discount(row):
        if row['start_day'] in weekdays:
            start_time = datetime.datetime.combine(datetime.date.today(), row['start_time'])
            if start_time.time() <= datetime.time(10, 0):
                factor = discount_factors['weekdays']['morning']
            elif start_time.time() <= datetime.time(18, 0):
                factor = discount_factors['weekdays']['day']
            else:
                factor = discount_factors['weekdays']['evening']
        else: 
            factor = discount_factors['weekends']
        
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            row[vehicle] *= factor
            
        return row

    df = df.apply(apply_discount, axis=1)
    
    return df


