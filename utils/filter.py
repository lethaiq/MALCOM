import glob
from models.robust import RobustFilter

class Filter:
    def __init__(self, dataset, topic_file):
        self.word2idx_dict, self.idx2word_dict = self.load_dict(dataset)
        self.topic_model = self.load_topic(topic_file)

    def load_topic(self, path):
        topic_model = None
        try:
            topic_model = TopicCoherency()
            topic_model.load(path)
        except Exception as e:
            print("ERROR loading topic model ", e)
        return topic_model

    def load_dict(self, dataset, vocab_size=-1, use_external_file=None):
        # print("loading dict with vocab_size", vocab_size)
        """Load dictionary from local files"""
        iw_path = 'dataset/{}_iw_dict.txt'.format(dataset)
        wi_path = 'dataset/{}_wi_dict.txt'.format(dataset)

        if not os.path.exists(iw_path) or not os.path.exists(iw_path):  # initialize dictionaries
            init_dict(dataset, vocab_size, use_external_file)

        with open(iw_path, 'r') as dictin:
            idx2word_dict = eval(dictin.read().strip())
        with open(wi_path, 'r') as dictin:
            word2idx_dict = eval(dictin.read().strip())
        return word2idx_dict, idx2word_dict


    def check(self, file):
        pass

    def run(self):
        rt = ""
        self.topic_model = self.load_topic(cfg.topic_file)
        self.robustChecker = RobustFilter(self.idx2word_dict, self.word2idx_dict, 
                                        self.topic_model, 
                                        spelling_thres=4,
                                        coherency_thres=0.15)
        files = glob.glob('./gen/*')
        for file in tqdm(files):
                
            s = "\nloaded {} | filterd out {}".format(file, g_loader.count_filter())
            rt += s
            print(s)

        print(rt)


robust_filter = Filter('gossipcopWithContent20x', './dataset/gossipcopWithContent20x_topic_7.pkl')
