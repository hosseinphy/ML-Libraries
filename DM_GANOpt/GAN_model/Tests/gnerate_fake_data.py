import json
import uuid 

print(uuid.uuid4())



dataset  = "/home/maps/DM_GANOpt/instances/ad052f6d-8db3-11f0-b83c-13157b2dc4a5/dataset.json"

with open(dataset, 'rb') as f:
    data = json.load(f)


complete = data.get('complete') or []

# print(complete)
uuids = []
if complete:
    for item in complete:
        if item.get('uuid'):
            uuids.append(item.get('uuid'))


print(uuids[-1:-17:-1])

