import json
import logging
import os
import pickle
from collections import defaultdict

from tools import text_parsing

PATH = os.path.abspath(os.path.dirname(__file__))

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DataHolder:
    def __new__(cls, use_cache=False):
        pickle_path = os.path.join(PATH, 'data_holder.pickle')
        if use_cache:
            if not os.path.exists(pickle_path):
                raise FileNotFoundError('Cached DataHolder does not exist, try set use_cache=False')

            with open(pickle_path, 'rb') as f:
                holder = pickle.load(f)
            return holder

        else:
            if os.path.exists(pickle_path):
                os.remove(pickle_path)
            return object.__new__(cls)

    def __init__(self, use_cache=False):
        self._data = dict()
        self._init_all_data()

        pickle_path = os.path.join(PATH, 'data_holder.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)

    # ========== Initial methods ==========

    def _init_all_data(self):
        self._init_languages()
        self._init_stopwords()
        self._init_letters()
        self._init_words()
        self._init_unigrams()
        self._init_trigrams()

    def _init_languages(self):
        logging.debug('_init_languages')
        with open(os.path.join(PATH, 'languages.json'), 'r') as f:
            self._data['languages'] = set(json.load(f))

    def _init_stopwords(self):
        logging.debug('_init_stopwords')
        if 'languages' not in self._data:
            self._init_languages()

        with open(os.path.join(PATH, 'stopwords.json'), 'r') as f:
            data = json.load(f)
            self._data['stopwords'] = {
                language: set(words) for language, words in data.items()
            }

    def _init_letters(self):
        logging.debug('_init_letters')
        if 'languages' not in self._data:
            self._init_languages()

        self._data['letters'] = dict()
        with open(os.path.join(PATH, 'alphabets.json'), 'r') as f:
            data = json.load(f)

        for language in self._data['languages']:
            d = dict()
            d['decapitalize'] = data[language]
            d['all'] = set(data[language].keys()) | set(data[language].values())
            self._data['letters'][language] = d

        self._data['letters']['all'] = set()
        self._data['letters']['decapitalize'] = dict()
        for language in self._data['languages']:
            self._data['letters']['all'] |= self._data['letters'][language]['all']
            self._data['letters']['decapitalize'].update(self._data['letters'][language]['decapitalize'])

        # is it necessary?
        self._data['letters']['all'].add('\'')

    def _init_words(self):
        logging.debug('_init_words')
        if 'languages' not in self._data:
            self._init_languages()
        if 'letters' not in self._data:
            self._init_letters()

        self._data['words'] = dict()
        for language in self._data['languages']:
            book_path = os.path.join(PATH, 'texts/{}.txt'.format(language))
            with open(book_path, 'r') as book:
                text = book.read()

            letters = self._data['letters'][language]['all']
            decapitalize = self._data['letters'][language]['decapitalize']

            self._data['words'][language] = text_parsing.get_words(text, letters, decapitalize)

        self._data['words']['all'] = set()
        for language in self._data['languages']:
            self._data['words']['all'] |= set(self._data['words'][language])

    def _init_unigrams(self):
        logging.debug('_init_unigrams')
        if 'letters' not in self._data:
            self._init_letters()
        if 'words' not in self._data:
            self._init_words()

        self._data['unigrams'] = dict()
        for language in self._data['languages']:
            self._data['unigrams'][language] = text_parsing.get_unigram_counts(self._data['words'][language])

        self._data['unigrams']['all'] = self._data['letters']['all']

    def _init_trigrams(self):
        logging.debug('_init_trigrams')
        if 'words' not in self._data:
            self._init_words()

        self._data['trigrams'] = dict()
        for language in self._data['languages']:
            self._data['trigrams'][language] = text_parsing.get_trigram_counts(self._data['words'][language])

        self._data['trigrams']['all'] = set()
        for language in self._data['languages']:
            self._data['trigrams']['all'] |= set(self._data['trigrams'][language].keys())

    # ========== Public methods ==========

    def get_languages(self):  # list[str]
        return sorted(list(self._data['languages']))

    def get_all_unigrams(self):  # list[str]
        return sorted(list(self._data['unigrams']['all']))

    def get_unigrams(self, language):  # defaultdict{str: int}
        return self._data['unigrams'][language]

    def get_all_letters(self):  # set[str]
        return self._data['letters']['all']

    def get_decapitalize(self):  # dict{str: str}
        return self._data['letters']['decapitalize']

    def get_all_trigrams(self):  # list[str]
        return sorted(list(self._data['trigrams']['all']))

    def get_trigrams(self, language):  # defaultdict{str: int}
        return self._data['trigrams'][language]

    def get_all_stopwords(self):  # list[str]
        all_stopwords = set()
        for _, words in self._data['stopwords'].items():
            all_stopwords |= words
        return sorted(list(all_stopwords))

    def get_stopwords(self, language):  # set[str]
        return self._data['stopwords'][language]
