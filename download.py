import os
import sys
import socket
import urllib.request

import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from torchvision.datasets.folder import IMG_EXTENSIONS

# decrease wait time for each download
socket.setdefaulttimeout(15)

# resize image
t_512 = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])

# IDs and URLs
multimedia = pd.read_csv('multimedia.txt', delimiter = '\t')
# IDs and labels
dataset = pd.read_csv('NZ-Species.csv', delimiter = '\t')
# iterate through instances to download
for i, row in tqdm(multimedia.iterrows()):
    species_dir = 'dataset/' +  dataset.loc[dataset['gbifID'] == row['gbifID'], 'verbatimScientificName'].iat[0]
    filename = species_dir + '/' + str(i) + '_' + str(row['gbifID']) + '.' + row['format'].split('/')[-1]
    if not (filename.count('.') == 1 and filename.lower().endswith(IMG_EXTENSIONS)):
        print('\nSkipped', str(i), row['gbifID'], row['identifier'], flush=True)
        continue
    try:
        if not os.path.exists(species_dir):
            os.makedirs(species_dir)
        urllib.request.urlretrieve(row['identifier'], filename)
        try:
            im = Image.open(filename)
            im.convert('RGB')
            t_512(im).save(filename)
        except:
            print('\nRemoved incompatible', str(i), row['gbifID'], row['identifier'], flush=True)
            os.remove(filename)
    except:
        print('\nFailed to acquire', str(i), row['gbifID'], row['identifier'], flush=True)
