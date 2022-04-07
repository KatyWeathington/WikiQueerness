__author__ = 'aaroniidx'
# import LIWCDictionary as liwc
import liwc
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from empath import Empath


class Analyzer:

    def __init__(self, analyzer):
        if analyzer == 'LIWC':
            self.analyzer = analyzer
            self.liwc_parse, self.liwc_category_names = liwc.load_token_parser('LIWC2007_English100131.dic')
            # self.liwc_dic = liwc.LIWCDictionary("LIWC2007.cat")
            self.headers = ['funct', 'pronoun', 'ppron', 'i', 'we', 'you',
                            'shehe', 'they', 'ipron', 'article', 'verb',
                            'auxverb', 'past', 'present', 'future', 'adverb',
                            'preps', 'conj', 'negate', 'quant', 'number', 'swear',
                            'social', 'family', 'friend', 'humans', 'affect',
                            'posemo', 'negemo', 'anx', 'anger', 'sad', 'cogmech',
                            'insight', 'cause', 'discrep', 'tentat', 'certain',
                            'inhib', 'incl', 'excl', 'percept', 'see', 'hear',
                            'feel', 'bio', 'body', 'health', 'sexual', 'ingest',
                            'relativ', 'motion', 'space', 'time', 'work', 'achieve',
                            'leisure', 'home', 'money', 'relig', 'death', 'assent',
                            'nonfl', 'filler']
            self.headers = self.liwc_category_names
            # print(self.liwc_category_names)


        elif analyzer == 'Empath':
            self.analyzer = analyzer
            self.headers = ['help', 'office', 'dance', 'money', 'wedding', 'domestic_work', 'sleep', 'medical_emergency',
                            'cold', 'hate', 'cheerfulness', 'aggression', 'occupation', 'envy', 'anticipation', 'family',
                            'vacation', 'crime', 'attractive', 'masculine', 'prison', 'health', 'pride', 'dispute', 'nervousness',
                            'government', 'weakness', 'horror', 'swearing_terms', 'leisure', 'suffering', 'royalty', 'wealthy', 'tourism',
                            'furniture', 'school', 'magic', 'beach', 'journalism', 'morning', 'banking', 'social_media',
                            'exercise', 'night', 'kill', 'blue_collar_job', 'art', 'ridicule', 'play', 'computer', 'college',
                            'optimism', 'stealing', 'real_estate', 'home', 'divine', 'sexual', 'fear', 'irritability',
                            'superhero', 'business', 'driving', 'pet', 'childish', 'cooking', 'exasperation', 'religion',
                            'hipster', 'internet', 'surprise', 'reading', 'worship', 'leader', 'independence', 'movement',
                            'body', 'noise', 'eating', 'medieval', 'zest', 'confusion', 'water', 'sports', 'death', 'healing',
                            'legend', 'heroic', 'celebration', 'restaurant', 'violence', 'programming', 'dominant_heirarchical',
                            'military', 'neglect', 'swimming', 'exotic', 'love', 'hiking', 'communication', 'hearing', 'order',
                            'sympathy', 'hygiene', 'weather', 'anonymity', 'trust', 'ancient', 'deception', 'fabric', 'air_travel',
                            'fight', 'dominant_personality', 'music', 'vehicle', 'politeness', 'toy', 'farming', 'meeting',
                            'war', 'speaking', 'listen', 'urban', 'shopping', 'disgust', 'fire', 'tool', 'phone', 'gain',
                            'sound', 'injury', 'sailing', 'rage', 'science', 'work', 'appearance', 'valuable', 'warmth',
                            'youth', 'sadness', 'fun', 'emotional', 'joy', 'affection', 'traveling', 'fashion', 'ugliness',
                            'lust', 'shame', 'torment', 'economics', 'anger', 'politics', 'ship', 'clothing', 'car',
                            'strength', 'technology', 'breaking', 'shape_and_size', 'power', 'white_collar_job', 'animal',
                            'party', 'terrorism', 'smell', 'disappointment', 'poor', 'plant', 'pain', 'beauty', 'timidity',
                            'philosophy', 'negotiate', 'negative_emotion', 'cleaning', 'messaging', 'competing', 'law',
                            'friends', 'payment', 'achievement', 'alcohol', 'liquid', 'feminine', 'weapon', 'children',
                            'monster', 'ocean', 'giving', 'contentment', 'writing', 'rural', 'positive_emotion', 'musical']

        elif analyzer == 'Custom Dictonary':
            self.analyzer = 'Custom Dictionary'

        elif analyzer == 'Vader':
            '''
            Details about running VADER can be found here:
            http://www.nltk.org/howto/sentiment.html
            '''
            ## Uncomment the following lines if you get errors about not having
            ## the correct NLTK files downloaded
            # nltk.download('vader_lexicon')
            # nltk.download('stopwords')
            self.analyzer = 'Vader'
            self.headers = ['compound', 'neg', 'neu', 'pos']
            self.sid = SentimentIntensityAnalyzer()

        else:
            raise NameError('Undefined analyzer.')

    def analyze(self, msg, dictionary=[]):
        if self.analyzer == 'LIWC':
            return self.liwc_analyze(msg)
        elif self.analyzer == 'Empath':
            return self.empath_analyze(msg)
        elif self.analyzer == 'Custom Dictionary':
            return self.dictionary_count_analyze(msg, dictionary)
        elif self.analyzer == 'Vader':
            return self.vader_analyze(msg)
        else:
            raise NameError('Undefined analyzer.')

    def liwc_tokenize(self, msg):
        # you may want to use a smarter tokenizer
        for match in re.finditer(r'\w+', msg, re.UNICODE):
            yield match.group(0)

    def liwc_counts(self, msg_tokens):
        from collections import Counter
        return Counter(category for token in msg_tokens for category in self.liwc_parse(token))

    def liwc_analyze_old(self, msg):
        # if len(msg) == 0:
        #     return {item: 0.0 for item in self.headers}

        msg = msg.lower()
        print(msg)
        tokens = self.liwc_tokenize(msg)
        print(tokens)
        # liwc_metrics = self.liwc_dic.get_counts(msg)
        wc = len(list(tokens))
        print(wc)
        counts = self.liwc_counts(tokens)
        print(counts)

        liwc_metrics = {item: 0.0 for item in self.headers}
        for item, count in counts:
            liwc_metrics[item] = count/wc
            print(item)
            print(count)
            print(wc)
        return liwc_metrics

    def liwc_analyze(self, msg):
        parse, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')

        msg = msg.lower()
        print(msg)

        tokens = self.liwc_tokenize(msg)
        print(tokens)

        wc = len(list(tokens))
        print(wc)

        from collections import Counter
        counts = Counter(category for token in tokens for category in parse(token))
        print(counts)

    def empath_analyze(self, msg):
        lexicon = Empath()
        result = lexicon.analyze(msg, normalize=True)
        if result:
            return result
        return {item: 0.0 for item in self.headers}

    def vader_analyze(self, msg):
        ss = self.sid.polarity_scores(msg)
        vader_metrics = {}
        for k in sorted(ss):
            vader_metrics[k] = ss[k]
        return vader_metrics

    def dictionary_count_analyze(self, msg, dictionary):
        count = 0
        words = nltk.Text(msg)
        for word in dictionary:
            count += words.count(word)
        return count
