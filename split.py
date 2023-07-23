import os
import tqdm
import pickle
import random
import shutil

random.seed(0)
classes = os.listdir('dataset')
os.makedirs('dataset/train')
os.makedirs('dataset/test')
instance_count = {}
for class_i in tqdm.tqdm(classes):
  image_files = os.listdir(f'dataset/{class_i}')
  instances = {}
  for image_file in image_files:
    instance_id = image_file.split('.')[0].split('_')[1]
    if instance_id in instances:
      instances[instance_id].append(image_file)
    else:
      instances[instance_id] = [image_file]
  num = len(instances)
  instance_ids = list(instances.keys())
  shutil.move(f'dataset/{class_i}', f'dataset/train/{class_i}')
  os.makedirs(f'dataset/test/{class_i}')
  instance_count_i = len(instance_ids)
  if num > 1:
    random.shuffle(instance_ids)
    for instance_id in instance_ids[::10]:
      for image_file in instances[instance_id]:
        os.rename(f'dataset/train/{class_i}/{image_file}', f'dataset/test/{class_i}/{image_file}')
      instance_count_i -= 1
  instance_count[class_i] = instance_count_i
pickle.dump(instance_count, open('instance_count.pkl', 'wb'))
