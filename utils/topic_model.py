from scipy.spatial.distance import cosine
import pickle
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy.stats import entropy

class TopicCoherency:
    def __init__(self, nmf=None, tfidf_vectorizer=None, path=None, start_index=0):
        self.nmf = nmf
        self.tfidf_vectorizer = tfidf_vectorizer
        self.start_index = start_index
        self.id2topics = {}
        if path:
            self.load(path)
    
    def get_num_topic(self):
        return len(self.nmf.components_)

    def get_topic_from_id(self, _id):
        return np.argmax(self.id2topics[_id])

    def get_top_words(self, want_topic, n_top_words):
        words = []
        feature_names = self.tfidf_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(self.nmf.components_):
            if topic_idx == want_topic:
                words = [feature_names[i]for i in topic.argsort()[:-n_top_words - 1:-1]]
        return words
    
    def load(self, path, verbose=True):
        self.nmf, self.tfidf_vectorizer, self.id2topics = pickle.load(open(path, 'rb'))
        if verbose:
            print("NMF with {} topics".format(self.nmf.components_.shape[0]))
        self.num_topics = self.nmf.components_.shape[0]
    
    def save(self, path):
        pickle.dump([self.nmf, self.tfidf_vectorizer, self.id2topics], open(path, 'wb'))
        print("saved")
        
    def make_id2topics_from_data(self, data, indices=None):
        contents = []
        ids = []
        if not indices:
            indices = range(len(data))
        for i in indices:
            ids.append(data[i]['id'])
            contents.append(data[i]['content'])
        tfidf = self.tfidf_vectorizer.transform(contents)
        topics = self.nmf.transform(tfidf)[:,self.start_index:]
        topics_int = np.flip(np.argsort(topics,1),1)[:,0]
        self.id2topics = {}
        for i in range(len(ids)):
            self.id2topics[ids[i]] = topics[i]
        self.topics_int = topics_int
        self.topics = topics

    def cal_coherency_from_ids(self, comments, ids, mean=True):
        topics = [self.id2topics[ids[i]] for i in range(len(ids))]
        return self.cal_coherency(comments, topics, mean)
    
    def cal_coherency_from_single_id(self, comments, _id, mean=False):
        def matrix_cosine_single(x, y):
            rt = np.array([cosine(x[i]+1e-5,y+1e-5) for i in range(len(x))])
            return rt
        
        tfidf = self.tfidf_vectorizer.transform(comments)
        topics_comments = self.nmf.transform(tfidf)[:,self.start_index:]
        sims = matrix_cosine_single(topics_comments, self.id2topics[_id])

        if mean:
            return 1 - np.mean(sims)
        return 1 - sims
    
    def cal_coherency(self, comments, topics, mean=True):
        def matrix_cosine(x, y):
            rt = np.array([cosine(x[i]+1e-5,y[i]+1e-5) for i in range(len(x))])
            return rt

        tfidf = self.tfidf_vectorizer.transform(comments)
        topics_comments = self.nmf.transform(tfidf)[:,self.start_index:]
        sims = matrix_cosine(topics_comments, topics)
        
        if mean:
            return 1 - sims.mean()
        else:
            return 1 - sims