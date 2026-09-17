"""FastF1 extraction and 10 Hz per-driver race telemetry resampling."""

from __future__ import annotations

from pathlib import Path
from .sensors import Sensors
import pickle


def _resample_car_data(car_data):
    """Return valid car data resampled without combining separate drivers."""
    frame = car_data.copy().dropna()
    #.loc[:, required] accesses all the rows keeping the data for the required(list defined above) columns
    #loc[col], loc[row] can work for accessing a row or column, for multiple rows you shoud do loc[[r1, r2, ...]]
    """
    - .copy.dropna(...), makes a copy of the data frame so that orig doesn't get affected
    - dropna removes rows by default if no indication is given of what to remove, and
    it removes them if any cell in that row contains a missing value(things like np.na)
    - Here we only consider if it has missing values in the columns given by subset but if not all columns considered
    """
    frame = frame[(frame["Throttle"] <= 100) & (frame["Throttle"] >= 0) & (frame["nGear"] >= 0)]#throttle is a percentage
    """
    - The (frame["Throttle"] <= 100) generates a boolean vector and the &s work column wise 
    and at the end frame contains all the rows which have a 1 in them in the 
    boolean vector 
    """
    frame = frame.drop_duplicates(subset="Date").sort_values("Date")
    #The index is the unique identifier of the row, so here you make the Date(a name of a column) the index
    #It's much faster to find values in the indexed column compared to a normal column
    #If no index set there will be a unique number given to each index
    if frame.empty:#axes of length 0
        return frame
    frame["DeltaTime"] = (frame["Date"].diff()).dt.total_seconds()
    frame = frame.iloc[1:]#This is to just deal with the NaN that is produced from the diff
    #I can't think of any good default value to give it so I thought I would clip the whole row
    
    #fixing types
    dTypeDict = {}
        
    return frame.drop(columns=["Date"], errors="ignore")

def keepRequiredColumns(car_data):
    required = ["Date"] + Sensors.SENSOR_NAME_COLUMNS#feature columns is a tuple of the info we need for autoencoder(rpm, throttle, ...)
    missing = set(required).difference(car_data.columns)
    if missing:
        raise ValueError(f"FastF1 car data is missing columns: {sorted(missing)}")
    return car_data[required]

def extract_2024_races(output: Path, cache_dir: Path, *, year: int = 2024,
                       max_sessions: int | None = None) -> int:
    """Download completed race telemetry for every listed driver and write CSV.

    FastF1 data is cached locally, so rerunning safely reuses downloaded session
    data. Individual unavailable sessions or drivers are skipped with a message.
    """
    try:
        import fastf1
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("install requirements.txt before extracting FastF1 telemetry") from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    #I don't think caching is really necessary because we create the data only once
    #we should never be making repeated calls as our detector trains on the data created
    fastf1.Cache.enable_cache(str(cache_dir))
    schedule = fastf1.get_event_schedule(year, include_testing=False)#include_testing is testing sessions of races
    #EventSchedule object and it is a datafram, you can access info about different events from here
    rows = []
    completed = 0#number of completed sessions
    finish = False
    sorted_segment_ids = []#this will hold the segment_ids sorted by the finishing position in the race
    for _, event in schedule.iterrows():
        try:
            round_number = int(event["RoundNumber"])
        except (KeyError, TypeError, ValueError):
            continue
        if round_number <= 0:
            continue
        else:
            finish = True
        if max_sessions is not None and completed >= max_sessions:
            break
        try:
            session = fastf1.get_session(year, round_number, "R")# There are many sessions in one round(practice, qualifying) -> R is for race, main race(final one after all the qualifiers)
            #round is basically like a grandprix
            session.load(telemetry=True, weather=False, messages=False)
            #before you access any data from session so you do something like session.laps or something you need to do session.load()
        except Exception as exc:  # FastF1 availability varies by session
            print(f"Skipping round {round_number}: {exc}")
            continue
        completed += 1
        for driver in session.drivers:#session drivers is sorted by finishing position
            try:
                laps = session.laps.pick_drivers(driver)
                segment = _resample_car_data(keepRequiredColumns(laps.get_car_data()))#ONLY DATA BY SENSOR_NAME_COLUMNS is given
            except Exception as exc:
                print(f"Skipping {event['EventName']} driver {driver}: {exc}")
                continue
            if len(segment) < 12:
                continue
            segment_id = f"{year}-R{round_number:02d}-{driver}"
            sorted_segment_ids.append(segment_id)
            segment["segment_id"] = segment_id
            rows.append(segment)
        if finish:
            break
    
    if not rows:
        raise RuntimeError("no valid FastF1 telemetry was extracted")
    output.parent.mkdir(parents=True, exist_ok=True)#if parent doesn't exist it creates, else nothing
    #parents is that any other ancestors up the tree also get created, 
    #exist_ok no exception if directory already exists
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output, index=False)
    output.cwd
    with open(output.parent/ "sorted_segment_ids.pkl", "wb") as f:
        pickle.dump(sorted_segment_ids, f)
    return len(result)
