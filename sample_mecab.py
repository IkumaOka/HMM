import MeCab

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
words = set(words)

# 単語：idの辞書を作成
words_to_idx = {}
for i, word in enumerate(words):
    words_to_idx[word] = i

