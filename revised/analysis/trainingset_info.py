import numpy as np

data =  np.load('/Users/hhyu/dev/team/suman_data/onh_mac_vft_data.npz')

ids = data['subject_id']

ids_unique = np.unique(ids[:, 0])
print('unique ids:', ids_unique.shape[0])
print('number of sequences in trainingset:', ids.shape[0])

np.savetxt('trainingset_unique_ids.csv', ids_unique, delimiter=",", fmt="%i")

age = data['age_at_visit_date_months']

intervals = [age[:,1]-age[:,0], age[:,2]-age[:,1], age[:,3]-age[:,2]]
intervals = np.concatenate(intervals)
print(intervals.shape)
print("mean intervisit interval (months)=", intervals.mean())
print("standard deviation of intervisit interval (months)=", intervals.std())
