import MeCab
import numpy as np
from hmmlearn import hmm

mecab = MeCab.Tagger()

input_dir = "./datasets/livedoor/dokujo-tsushin/"
input_data = input_dir + "dokujo-tsushin-4778030.txt"


with open(input_data) as f:
    text = f.read()
    mecab.parse('') # 文字列がGCされるのを防ぐ
    node = mecab.parseToNode(text)
    words = []
    while node:
        # 単語を取得
        word = node.surface
        words.append(word)
        # 次の単語に進める
        node = node.next
# リスト内の単語の重複を除去
set_words = set(words)

# 単語：idの単語辞書を作成
words_id = {}
for i, word in enumerate(set_words):
    words_id[word] = i

# 単語系列を全てwords_idのidに変換
words_to_id = []
for word in words:
    for key in words_id:
        # print(key)
        if word == key:
            words_to_id.append(words_id[key])

X = np.reshape(words_to_id, [len(words_to_id), 1])

# verbose=Trueで各回のイテレーションを確認できる．
model = hmm.MultinomialHMM(n_components=10, n_iter=1000, verbose=True)

model.fit(X)

L,Z = model.decode(X)
# print(model.transmat_) # 遷移確率の出力 
# print(model.monitor_) # historyの配列は最後から2つの対数尤度を出力している．