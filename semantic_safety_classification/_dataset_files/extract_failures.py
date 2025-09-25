import pickle
import sys

with open('failures'+sys.argv[2]+'.pkl', 'rb') as f:
    failures = pickle.load(f)

count = int(sys.argv[1])

with open('failures.pkl', 'wb') as f:
    pickle.dump(failures[:count], f)
