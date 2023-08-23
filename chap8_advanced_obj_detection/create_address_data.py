

#%%
import pandas as pd

data_json = {'id': [1,2,3,4],
             'location_id': [1,2,3,4],
             'address_1': ['Middlefield Road', '24 Second Avenue', 
                           '24 Second Avenue', '24 Second Avenue'
                           ]
             }

addresses = pd.DataFrame(data_json)

#%%
addresses.to_csv('addresses.csv')
# %%
