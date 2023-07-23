import os
import tqdm
import shutil

import pandas as pd

def format_row(row):
  if row['taxonRank'] in ['SUBSPECIES', 'VARIETY', 'FORM']:
    return (row['species'], row['taxonRank'], row['infraspecificEpithet'])
  return (row[row['taxonRank'].lower()], row['taxonRank'])

def move_class(source_name, target_name):
  if source_name != target_name:
    current_classes = set(os.listdir('dataset/'))
    if source_name in current_classes:
      if target_name in current_classes:
        for file_i in os.listdir(f'dataset/{source_name}'):
          shutil.move(f'dataset/{source_name}/{file_i}', f'dataset/{target_name}/{file_i}')
        os.rmdir(f'dataset/{source_name}')
      else:
        shutil.move(f'dataset/{source_name}', f'dataset/{target_name}')
    else:
      print(f'Rename class nonexistent {source_name},{target_name}', flush=True)

info = pd.read_csv('NZ-Species.csv', delimiter = '\t')
classes = {}
print('Preprocessing classes', flush=True)
for class_name in tqdm.tqdm(os.listdir('dataset/')):
  row = info[info['verbatimScientificName'] == class_name].iloc[0]
  classes[class_name] = format_row(row)
new_classes = {}
print('Sanitising classes', flush=True)
for k, v in tqdm.tqdm(classes.items()):
  if v[1] in ['GENUS', 'ORDER', 'FAMILY']:
    if ' ' not in k or k.endswith('st1') or k.endswith('st2') or k.endswith('virus'):
      new_classes[k] = None
    else:
      new_classes[k] = k
  elif v[1] == 'SPECIES':
    if k.endswith('virus') or ' ' not in k:
      new_classes[k] = None
    elif k.split(' ')[0 : 2] == v[0].split(' ') or v[0] in classes:
      new_classes[k] = v[0]
    elif len(k.split(' ')) == 3 and ' '.join(k.split(' ')[0 : 2]) in classes:
      new_classes[k] = ' '.join(k.split(' ')[0 : 2])
    else:
      new_classes[k] = k
  else:
    new_classes[k] = (' '.join(k.split(' ')[0 : 2]), v[0])
delete_classes = []
rename_classes = {}
for k, v in tqdm.tqdm(new_classes.items()):
  if type(v) is tuple:
    continue
  else:
    if v == None:
      delete_classes.append(k)
    elif v in rename_classes:
      rename_classes[v].append(k)
    else:
      rename_classes[v] = [k]
for k, v in tqdm.tqdm(new_classes.items()):
  if type(v) is tuple:
    if k == 'Penion cuvierianus jeakingsi':
      rename_classes['Penion ormesi'] = [k]
    elif v[0] in rename_classes:
      rename_classes[v[0]].append(k)
    elif v[1] in rename_classes:
      rename_classes[v[1]].append(k)
    else:
      rename_classes[v[0]] = [k]
print('Processing classes', flush=True)
for delete_class in tqdm.tqdm(sorted(delete_classes)):
  shutil.rmtree(f'dataset/{delete_class}/')
flagged_classes = []
for k, v in tqdm.tqdm(rename_classes.items()):
  if len(v) == 1:
    if v[0] == k and len(k.split(' ')) != 2 and ' × ' not in k and ' x ' not in k:
      flagged_classes.append(v[0])
    else:
      move_class(v[0], k)
  else:
    for vi in v:
      if vi.startswith(k):
        move_class(vi, k)
      else:
        flagged_classes.append(vi)

print('Carrying out special instructions', flush=True)
special_instructions = open('special_instructions.txt', 'r').read().split('\n')
for special_i in tqdm.tqdm(special_instructions):
  if ',' in special_i:
    parsed_i = special_i.split(',')
    if parsed_i[0] == 'D':
      if parsed_i[1] in os.listdir('dataset/'):
        shutil.rmtree(f'dataset/{parsed_i[1]}')
        if parsed_i[1] in flagged_classes:
          flagged_classes.remove(parsed_i[1])
        else:
          print(f'Delete class not flagged {parsed_i[1]}', flush=True)
      else:
        print(f'Delete class nonexistent {parsed_i[1]}', flush=True)
    elif parsed_i[0] == 'K':
      if parsed_i[1] in flagged_classes:
        flagged_classes.remove(parsed_i[1])
      else:
        print(f'Keep class not flagged {parsed_i[1]}', flush=True)
    elif parsed_i[0] == 'R':
      for vi in parsed_i[2:]:
        move_class(vi, parsed_i[1])
        if vi in flagged_classes:
          flagged_classes.remove(vi)
        else:
          print(f'Rename class not flagged {vi}', flush=True)
  elif special_i != '':
    print(f'Unexpected instruction {special_i}', flush=True)

print(f'Unresolved flagged classes: {flagged_classes}', flush=True)

print('Removing empty classes', flush=True)
for class_name in tqdm.tqdm(os.listdir('dataset/')):
  if not os.listdir(f'dataset/{class_name}/'):
    os.rmdir(f'dataset/{class_name}/')
