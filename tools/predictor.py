import logging
import os
import pickle
from sklearn.naive_bayes import MultinomialNB

from tools import text_parsing
from tools.data.data_holder import DataHolder

PATH = os.path.abspath(os.path.dirname(__file__))

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class Predictor:
    def __new__(cls, use_cache=False):
        pickle_path = os.path.join(PATH, 'predictor.pickle')
        if use_cache:
            if not os.path.exists(pickle_path):
                raise FileNotFoundError('Cached Predictor does not exist, try set use_cache=False')

            with open(pickle_path, 'rb') as f:
                predictor = pickle.load(f)
            return predictor

        else:
            return object.__new__(cls)

    def __init__(self, use_cache=False):
        self._data_holder = DataHolder(use_cache)

        # train all models
        self._init_unigram_bayes_method()
        self._init_trigram_bayes_method()
        self._init_stopwords_bayes_method()

        pickle_path = os.path.join(PATH, 'predictor.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)

    # ========== Training models ==========

    def _init_unigram_bayes_method(self):
        log.debug('collecting features for unigram bayes method')
        features = []
        languages = self._data_holder.get_languages()

        for language in languages:
            unigram_count = self._data_holder.get_unigrams(language)
            features.append(self._unigram_count_to_vector(unigram_count))

        log.debug('training unigram bayes method')
        self._unigram_bayes = MultinomialNB()
        self._unigram_bayes.fit(features, languages)

    def _init_trigram_bayes_method(self):
        log.debug('collecting features for trigram bayes method')
        features = []
        languages = self._data_holder.get_languages()

        for language in languages:
            trigram_count = self._data_holder.get_trigrams(language)
            features.append(self._trigram_count_to_vector(trigram_count))

        log.debug('training trigram bayes method')
        self._trigram_bayes = MultinomialNB()
        self._trigram_bayes.fit(features, languages)

    def _init_stopwords_bayes_method(self):
        log.debug('collecting features for stopwords bayes method')
        features = []
        languages = self._data_holder.get_languages()

        for language in languages:
            stopwords = self._data_holder.get_stopwords(language)
            features.append(self._stopwords_to_vector(stopwords))

        log.debug('training stopwords bayes method')
        self._stopwords_bayes = MultinomialNB()
        self._stopwords_bayes.fit(features, languages)

    # ========== Auxiliary methods ==========

    def _unigram_count_to_vector(self, unigram_count):  # list[int]
        all_unigrams = self._data_holder.get_all_unigrams()
        vector = []

        for unigram in all_unigrams:
            vector.append(unigram_count.get(unigram) or 0)

        return vector

    def _trigram_count_to_vector(self, trigram_count):
        all_trigrams = self._data_holder.get_all_trigrams()
        vector = []

        for trigram in all_trigrams:
            vector.append(trigram_count.get(trigram) or 0)

        return vector

    def _stopwords_to_vector(self, stopwords):
        all_stopwords = self._data_holder.get_all_stopwords()
        return [1 if word in stopwords else 0 for word in all_stopwords]

    # ========== Predict methods ==========

    def unigram_bayes_method(self, unigram_counts):
        log.debug('predicting with unigram bayes method')
        features = self._unigram_count_to_vector(unigram_counts)
        return self._unigram_bayes.predict([features])[0]

    def trigram_bayes_method(self, trigram_counts):
        log.debug('predicting with trigram bayes method')
        features = self._trigram_count_to_vector(trigram_counts)
        return self._trigram_bayes.predict([features])[0]

    def stopwords_bayes_method(self, stopwords):
        log.debug('predicting with stopwords bayes method')
        features = self._stopwords_to_vector(stopwords)
        return self._stopwords_bayes.predict([features])[0]

    # ========== Public methods ==========

    def predict(self, text):  # dict{str: str}
        words = text_parsing.get_words(
            text,
            self._data_holder.get_all_letters(),
            self._data_holder.get_decapitalize(),
        )

        unigram_counts = text_parsing.get_unigram_counts(words)
        trigram_counts = text_parsing.get_trigram_counts(words)

        return {
            'Unigram Bayes method': self.unigram_bayes_method(unigram_counts),
            'Trigram Bayes method': self.trigram_bayes_method(trigram_counts),
            'Stopwords Bayes method': self.stopwords_bayes_method(words),
        }
