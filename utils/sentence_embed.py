import pandas as pd
import tensorflow as tf 
import tensorflow_hub as hub
import sys
import numpy as np
from tqdm import tqdm
import os
import glob

tf.compat.v1.disable_eager_execution()

dataset = sys.argv[1]
df = pd.read_csv('dataset/{}_{}.csv'.format(dataset, sys.argv[3]))
messages = list(df['text'])
ids = np.array(list(df['id']))
bs = int(sys.argv[2])

print("total messages:", df.shape)

embed = hub.Module("tf_modules/")
embedings = []

with tf.compat.v1.Session()  as session: 
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()]) 
    j = 0
    for i in tqdm(range(0, len(messages), bs)):
        file = 'dataset/{}/{}_{}.npy'.format(dataset, j, sys.argv[3])
        if not os.path.exists(file):
            message_embeddings = session.run(embed(messages[i:i+bs]))
            np.save(file, message_embeddings)
        else:
            print("skiped", file)
        j+=1

# concatenate
print('loading all embeddings and concat')
all_embeddings = []
files = glob.glob('dataset/{}/*.npy'.format(dataset))
for i in range(j):
    file = 'dataset/{}/{}_{}.npy'.format(dataset, i, sys.argv[3])
    if file in files:
        all_embeddings.append(np.load(file))
    else:
        print("ERROR, cannot find", file)
all_embeddings = np.concatenate(all_embeddings, 0)

# put to dictionary
# embedding_dict = {}
# for i in range(len(all_embeddings)):
#     embedding_dict[ids[i]] = all_embeddings[i]

# save to dictionary
print('saving compressed data to disk')
np.savez('dataset/{}_{}_embeddings.npz'.format(dataset, sys.argv[3]), ids=ids, embeddings=all_embeddings)


