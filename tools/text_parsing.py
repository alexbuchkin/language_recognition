from collections import defaultdict


def get_words(text, letters, decapitalize):  # list[str]
    words = []
    current_word = []

    for c in text:
        if c in letters:
            c = decapitalize.get(c) or c
            current_word.append(c)

        else:
            if current_word:
                words.append(''.join(current_word))
            current_word = []

    return words


def get_unigram_counts(words):  # defaultdict{str: int}
    unigram_count = defaultdict(int)

    for word in words:
        for unigram in word:
            unigram_count[unigram] += 1

    return unigram_count


def get_trigram_counts(words):  # defaultdict{str: int}
    trigram_count = defaultdict(int)

    for word in words:
        if len(word) < 3:
            continue
        for i in range(len(word) - 2):
            trigram_count[word[i:i + 3]] += 1

    return trigram_count
