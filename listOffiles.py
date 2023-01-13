import os
import json
dict_names = {}
for path, subdirs, files in os.walk('MIR_DATASETS_B'):
    for name in files:
        print(os.path.join(path, name))
        dict_names[name] = os.path.join(path, name)

with open('dict_name.json', 'w') as f:
    json.dump(dict_names, f)
