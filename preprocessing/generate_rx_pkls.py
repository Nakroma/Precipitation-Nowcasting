import pandas as pd
import os
import datetime
import pickle


# Generates pkl that has all datetimes that correspond to an existing input image
def generate_valid_datetimes():
    valid_datetimes = []
    for year in os.listdir('./rx_data'):
        # Skip non relevant folders
        if year not in ['2017', '2018', '2019', '2020', '2021']:
            continue
        for month in os.listdir('./rx_data/' + year):
            for day in os.listdir('./rx_data/' + year + '/' + month):
                print(datetime.datetime(int(year), int(month), int(day)))
                for hour_raw in os.listdir('./rx_data/' + year + '/' + month + '/' + day):
                    # Read hour and minute from filename
                    hour = hour_raw[9:11]
                    minute = hour_raw[11:13]
                    valid_datetimes.append(datetime.datetime(int(year), int(month), int(day), int(hour), int(minute)))
    pickle.dump(frozenset(valid_datetimes), open('./rx_data/valid_datetime.pkl', 'wb'))


# Generates pkl's for separating days into train/valid/test sets
def generate_rainy_days():
    datetimes = []
    for year in os.listdir('./rx_data'):
        # Skip non relevant folders
        if year not in ['2017', '2018', '2019', '2020', '2021']:
            continue
        print(year)
        for month in os.listdir('./rx_data/' + year):
            print(month)
            for day in os.listdir('./rx_data/' + year + '/' + month):
                for hour in range(24):
                    for minute in range(0, 60, 5):
                        datetimes.append(datetime.datetime(int(year), int(month), int(day), hour, minute))

    # has_rainfall column isn't actually used so we just init it to 0
    df = pd.DataFrame(columns=['has_rainfall'], index=datetimes)
    df['has_rainfall'] = 0

    test = df[:(100 * 288)]
    train = df[(100 * 288):(825 * 288)]
    valid = df[(825 * 288):]

    train.to_pickle('./rx_data/pd/hko7_rainy_train.pkl', protocol=4)
    test.to_pickle('./rx_data/pd/hko7_rainy_test.pkl', protocol=4)
    valid.to_pickle('./rx_data/pd/hko7_rainy_valid.pkl', protocol=4)


generate_valid_datetimes()
generate_rainy_days()
