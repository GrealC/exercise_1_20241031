import pandas as pd
from collections import Counter

"""Task1.2"""
with open('text8.txt', 'r', encoding='utf-8') as file:
    text = file.read()

words = text.split()
word_count = Counter(words)

sorted_word_counts = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

vocabulary_dict = {word: word_id for word_id, (word, _) in enumerate(sorted_word_counts)}


words_transferred = []
for word in words:
    word_id = vocabulary_dict.get(word)
    if word_id is None:
        raise ValueError("out of vocabulary")
    words_transferred.append(word_id)

print(len(words_transferred))
