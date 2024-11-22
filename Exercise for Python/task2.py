
"""
distinct():
# take a string and number
# return a distinct-n score

1. 接收一个字符串，以及数字n
2. 将传入的字符串按照数字依次分为多个n串并放在列表中
3. 计算列表的总长度 l
4. 计算列表中元素出现的次数，并计算出现次数为1的元素个数 m
5. 计算distinct-n score = m / l

I like the following
4
2： 0 - 1 - 2
0-2
1-3
2-4
"""
from collections import Counter

def distinct(token: str, n: int)->float:
    words = token.split()
    ngrams = []

    # Split the input string into n-grams (n-grams are sequences of n consecutive words)
    for i in range(len(words) - n + 1):
        gram = words[i:i+n]
        ngrams.append(gram)

    # Calculate the total number of ngrams
    len_ngrams = len(ngrams)

    # Calculate the number of ngrams that appear only once
    m = sum(1 for ngram in ngrams if ngrams.count(ngram)==1)

    # Calculate distinct-n score
    distinct_n_score = m / len_ngrams

    return distinct_n_score

# s = "I like to read books. I enjoy reading novels."
# s = "I like to read books. I like reading novels"
#
# score1 = distinct(s, 1)
# print(score1)
#
# score2 = distinct(s, 2)
# print(score2)

with open("text8.txt", "r") as file:
    tokens = file.read()
    score1 = distinct(tokens, 1)
    score2 = distinct(tokens, 2)
    score3 = distinct(tokens, 3)
    score4 = distinct(tokens, 4)

    print(f"Distinct-1 score: {score1}")
    print(f"Distinct-2 score: {score2}")
    print(f"Distinct-3 score: {score3}")
    print(f"Distinct-4 score: {score4}")
