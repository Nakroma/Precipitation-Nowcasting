import wradlib as wrl
import numpy as np
from PIL import Image
import os
from pathlib import Path
import time
import pandas as pd

# Accumulate total rainfall. This is used to generate the pkl files needed by the model later
df = pd.DataFrame(columns=['total_rainfall'], index=pd.date_range('2017-01-01', '2021-12-31', freq='D'))
df['total_rainfall'] = 0

saved_day = ''
start = time.time()
for root, dirs, files in os.walk('/home/uni/rx'):
    for file in files:
        if not file.startswith('raa'):
            continue
        if file.endswith('.bz2'):
            continue
        datestring = file.split('-')[2]
        year = '20' + datestring[:2]
        month = datestring[2:4]
        day = datestring[4:6]
        minute = datestring[6:10]
        savestring = 'RAD' + datestring + '00'
        
        if day != saved_day:
            print('... proccesed in ' + str(time.time() - start) + 's')
            start = time.time()
            saved_day = day
            print('Processing year ' + year + ', month ' + month + ', day ' + day)

        # https://docs.wradlib.org/en/stable/notebooks/radolan/radolan_showcase.html
        data, metadata = wrl.io.read_radolan_composite(os.path.join(root, file))
        data = np.ma.masked_equal(data, -9999) / 2 - 32.5
        # Take 480px x 480px cut from the center
        cut = data[210:-210, 310:-110]

        # Convert dBZ to pixel values and write them into a BW image
        v = np.clip(np.ceil(255 * ((cut + 10.0) / 70.0) + 0.5), 0, 255).astype('uint8')
        im = Image.fromarray(np.transpose(v), 'L')
        im = im.rotate(90)

        # Calculate total rainfall and write it into dataframe
        cut[cut < 0] = 0
        r = np.power((np.power(10, cut / 10.0) / 200), 5/8)
        pd_idx = year + '-' + month + '-' + day
        df.at[pd_idx, 'total_rainfall'] = df.at[pd_idx, 'total_rainfall'] + np.sum(r)
        
        save_path = './rx_data/' + year + '/' + month + '/' + day
        Path(save_path).mkdir(parents=True, exist_ok=True)
        im.save(save_path + '/' + savestring + '.png')
        
df.to_pickle('./rx_data/total_rainfall.pkl')
