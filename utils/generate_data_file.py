import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import fastTest as fastText
import sys
reload(sys)
sys.setdefaultencoding('UTF8') # default encoding to utf-8
lid_model = fastText.load_model("/Users/lethai/Downloads/lid.176.ftz")


dataset = "gossipcop"
max_comments = 4
min_comment_words = 4
# max_comment_words = 20
min_comment_coherence_words = 5
max_coherence_comment = 2
max_comment = 5
max_num_d_comments = 100000


real = pd.read_csv('./{}_real.csv'.format(dataset))
fake = pd.read_csv('./{}_fake.csv'.format(dataset))
combine = pd.concat([real, fake], axis=0)

id2idx = {}
ids = combine['id'].get_values()
for i in range(len(ids)):
    assert ids[i] not in id2idx
    id2idx[ids[i]] = i

comment = pd.read_csv('./{}_comment_no_ignore.tsv'.format(dataset), sep='\t')
content = pd.read_csv('./{}_content_no_ignore.tsv'.format(dataset), sep='\t')
content_ids = content['id']
comment_ids = comment['id']
np.intersect1d(content_ids, comment_ids).shape

comment_id2idx = {}
for i in range(len(comment)):
    comment_id2idx[comment['id'][i]] = i
content_id2idx = {}
for i in range(len(content)):
    content_id2idx[content['id'][i]] = i

data = []
for i in range(len(content)):
    temp = {}
    curr_id = content['id'][i]
    if curr_id in id2idx:
        temp['title'] = combine['title'].array[id2idx[curr_id]].lower()
    else:
        print(curr_id)
        temp['title'] = None
    temp['content'] = content['content'][i].lower()
    temp['label'] = content['label'][i]
    com_str = comment['comment'][comment_id2idx[curr_id]].lower()
    coms = com_str.split('::')
    temp['comments'] = coms
    data.append(temp)

print(len(data))


# train D
X = []
for i in range(len(data)):
        for comment in data[i]['comments']:
            if len(comment.split())>= min_comment_words:
                X.append(comment)
np.random.shuffle(X)
# X = X[:max_num_d_comments]
total_train = int(len(X)*0.95)
X_train = X[:total_train]
X_test = X[total_train:]
with open('{}.txt'.format(dataset),'w') as f:
    for sent in X_train:
        f.write("{}\n".format(sent))
with open('{}_test.txt'.format(dataset),'w') as f:
    for sent in X_test:
        f.write("{}\n".format(sent))


import textdistance
def similarity(s1, s2):
    return textdistance.overlap.similarity(s1, s2)

# train H
y = []
X = []
for i in range(len(data)):
    num = 0
    for comment in data[i]['comments'][:max_coherence_comment]:
        if len(comment.split())>min_comment_coherence_words:
#                 f.write("{}::{}\n".format(data[i]['title'], comment))
            if similarity(data[i]['title'], comment) > 0.8:
#                 print(similarity(data[i]['title'], comment))
                X.append("{}::{}".format(data[i]['title'], comment))
                y.append(1)
                num += 1
        
    for j in range(len(data[i]['comments'][:max_coherence_comment])):
        idx = i
        while idx == i:
            idx = np.random.choice(len(data))
        for comment in data[idx]['comments'][:num]:
            if len(comment.split())>min_comment_coherence_words:
                X.append("{}::{}".format(data[i]['title'], comment))
                y.append(0)
                break 
X = np.array(X)
y = np.array(y)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
with open('{}_coherence_train_data.txt'.format(dataset), 'w') as f:
    for sent in X_train:
        f.write("{}\n".format(sent))
with open('{}_coherence_train_label.txt'.format(dataset), 'w') as f:
    for label in y_train:
        f.write("{}\n".format(label))
with open('{}_coherence_test_data.txt'.format(dataset), 'w') as f:
    for sent in X_test:
        f.write("{}\n".format(sent))
with open('{}_coherence_test_label.txt'.format(dataset), 'w') as f:
    for label in y_test:
        f.write("{}\n".format(label))


# import pandas as pd
# allrows = []
# for i in range(len(X)):
#     tmp = {}
#     s = X[i].split('::')
#     tmp['text1'] = s[0]
#     tmp['text2'] = s[1]
#     tmp['similarity'] = y[i]
#     allrows.append(tmp)
# df = pd.DataFrame.from_dict(allrows)
# df.to_csv('{}_coherence_ludwig.csv'.format(dataset))


# train F
y = []
X = []
for i in range(len(data)):
        num = 0
        comments = []
        for comment in data[i]['comments']:
            if len(comment.split())>min_comment_words:
                comments.append(comment)
                num += 1
            if num >= max_comment:
                break
        X.append("{}::{}".format(data[i]['title'], "::".join(comments)))
        y.append(data[i]['label'])

X = np.array(X)
y = np.array(y)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
with open('{}_detect_train_data.txt'.format(dataset),'w') as f:
    for sent in X_train:
        f.write("{}\n".format(sent))
with open('{}_detect_train_label.txt'.format(dataset),'w') as f:
    for label in y_train:
        f.write("{}\n".format(label))
with open('{}_detect_test_data.txt'.format(dataset),'w') as f:
    for sent in X_test:
        f.write("{}\n".format(sent))
with open('{}_detect_test_label.txt'.format(dataset),'w') as f:
    for label in y_test:
        f.write("{}\n".format(label))


print((len(y_train), len(y_test), np.unique(y, return_counts=True)))

# # ALL
# """ gossipcop_all.txt
#     title + comment1 + comment2 + comment3
#     title + comment1 + comment2 + comment3
# """

# TRAIN F
""" gossipcop_detect.txt
    title::comment1::comment2::comment3
    title::comment1::comment2
    title::comment1
"""

""" gossipcop_detect_label.txt
    1
    1
    0
"""

# TRAIN D
""" gossipcop_comment.txt
    comment1
    comment2
    comment3
    comment4
    ...
    commentN
"""

# TRAIN H
""" gossipcop_coherence.txt
    title::comment
    title::comment
    title::comment
"""

""" gossipcop_coherence_label.txt
    1
    0
    1
"""

