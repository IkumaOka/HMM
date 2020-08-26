import MeCab
import numpy as np

class TextAnalysis():
    def mecab_analysis(input_data):
        mecab = MeCab.Tagger()
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

        return X
