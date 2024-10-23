from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    
    for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)):  
            group.append(lst[i + j])
        result.extend(group[::-1])  

    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    
    return dict(sorted(result.items()))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(path, available):
        if not available:
            result.append(path[:])
            return
        
        for i in range(len(available)):
            if i > 0 and available[i] == available[i - 1]:
                continue
            backtrack(path + [available[i]], available[:i] + available[i+1:])
    
    nums.sort()  # Sort to easily skip duplicates
    result = []
    backtrack([], nums)
    return result


import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_pattern = r'(\d{2}-\d{2}-\d{4})|(\d{2}/\d{2}/\d{4})|(\d{4}\.\d{2}\.\d{2})'
    matches = re.findall(date_pattern, text)
    dates = [match for group in matches for match in group if match]
    
    return dates


import polyline
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    df['distance'] = 0.0

    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
        lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
        df.at[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df



def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all elements in the same row and column, excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    rotated_matrix = [[matrix[n - 1 - j][i] for j in range(n)] for i in range(n)]
    transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum
    
    return transformed_matrix



import pandas as pd
from pandas import Timestamp
import numpy as np

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) 
    pair cover a full 24-hour period for all 7 days of the week (Monday to Sunday).
    
    Args:
        df (pd.DataFrame): A DataFrame with columns 'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'.
    
    Returns:
        pd.Series: A boolean series with multi-index (id, id_2) indicating whether the data is incorrect.
    """

    df['startDay'] = pd.to_datetime(df['startDay']).dt.day_name()
    df['endDay'] = pd.to_datetime(df['endDay']).dt.day_name()

    full_week = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    full_day_hours = pd.date_range('00:00', '23:59', freq='T').time 

    def check_completeness(group):
        unique_days = set(group['startDay'].unique()).union(set(group['endDay'].unique()))
        if unique_days != full_week:
            return True  

        for day in full_week:
            day_entries = group[(group['startDay'] == day) | (group['endDay'] == day)]
            
            if not day_entries.empty:
                covered_minutes = set()
                for _, row in day_entries.iterrows():
                    start_time = Timestamp(row['startTime']).time()
                    end_time = Timestamp(row['endTime']).time()
                    time_range = pd.date_range(start=start_time, end=end_time, freq='T').time
                    covered_minutes.update(time_range)
                
                if len(covered_minutes) != len(full_day_hours):
                    return True  
        return False  

    result = df.groupby(['id', 'id_2']).apply(check_completeness)
    
    return result


