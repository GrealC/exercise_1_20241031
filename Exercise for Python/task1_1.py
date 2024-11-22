import pandas as pd
from collections import Counter

"""Task1.1"""
with open('text8.txt', 'r', encoding='utf-8') as file:
    text = file.read()

words = text.split()
word_count = Counter(words)

print(word_count)

sorted_word_counts = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

vocabulary = [(word_id, word, freq) for word_id, (word, freq) in enumerate(sorted_word_counts)]

df = pd.DataFrame(vocabulary, columns=['Word ID', 'Word', 'Frequency'])

print(df.head(10))

with open('vocabulary.txt', 'w', encoding='utf-8') as file:
    for word_id, word, freq in vocabulary:
        file.write(f'{word_id}\t{word}\t{freq}\n')

