import random
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer

tokenizer = BertTokenizer.from_pretrained('uncased_L-12_H-768_A-12/vocab.txt')
basic_tokenizer = BasicTokenizer()

with open('ED-data/Few-Shot_ED.json', 'r') as f:
    data = json.loads(f.read())
# test1 Movement 5 labels
# test2 Conflict 4 labels
# test3 Life     10 labels
# test4 Sports   4 labels
# test5 Business 12 labels
# test6 Military 4 labels
# test7 Music    5 labels
domains = ['Movement', 'Conflict', 'Life', 'Sports', 'Business', 'Military', 'Music']

for order in range(0, 7):
    print('order', order)
    test = {}
    for test_order in range(0, 7):
        if test_order == order or test_order == (order + 1) % 7:
            continue
        B_label = domains[test_order]
        print(B_label)
        L_labels = []

        # select Movement.*
        domain = {}
        for i in list(data.keys()):
            if i.split('.')[0] == B_label:
                L_labels.append(i)
                domain[i.split('.')[1]] = data[i]

        # add to sentences
        sentences = []
        for k, v in domain.items():
            for i in v:
                i.append(k)
                if i not in sentences:
                    sentences.append(i)

        print(len(sentences))

        test[B_label] = []
        support = []
        batch = []

        # total 200
        for n in range(0, 100):
            episode = {'support': {}, 'batch': {}}
            # First shuffle
            support_examples = []
            batch_examples = []
            random.shuffle(sentences)
            while len(batch_examples) < 20:
                t = random.sample(sentences, 1)[0]
                if t[1] in t[0].split(' ') and t not in batch_examples:
                    batch_examples.append(t)
            for k, v in domain.items():
                t = random.sample(v, 1)[0]
                while t in batch_examples or t[1] not in t[0].split(' '):
                    t = random.sample(v, 1)[0]
                support_examples.append(t)

            support_data = {'seq_ins': [], 'labels': [], 'seq_outs': [], 'word_piece_marks': [], 'tokenized_texts': [],
                            'word_piece_labels': []}
            for stn in support_examples:
                raw_data = stn
                ins = basic_tokenizer.tokenize(raw_data[0])
                support_data['seq_ins'].append(ins)

                support_data['labels'].append(B_label)

                outs = ['O'] * len(ins)
                keyword = basic_tokenizer.tokenize(raw_data[1])
                try:
                    outs[ins.index(keyword[0])] = "B-" + raw_data[3]
                except ValueError:
                    print('support')
                    print(raw_data)
                    continue
                if len(keyword) > 1:
                    for k in range(1, len(keyword)):
                        outs[ins.index(keyword[0]) + k] = "I-" + raw_data[3]
                support_data['seq_outs'].append(outs)

                texts = tokenizer.tokenize(raw_data[0])
                support_data['tokenized_texts'].append(texts)

                piece_marks = [0] * len(texts)
                for i in range(0, len(texts)):
                    if texts[i][:2] == '##':
                        piece_marks[i] = 1
                support_data['word_piece_marks'].append(piece_marks)

                piece_labels = ['O'] * len(piece_marks)
                if keyword[0] in texts:
                    piece_labels[texts.index(keyword[0])] = "B-" + raw_data[3]
                if len(keyword) > 1:
                    for k in range(1, len(keyword)):
                        if keyword[k] in texts:
                            piece_labels[texts.index(keyword[k])] = "I-" + raw_data[3]
                        outs[ins.index(keyword[0]) + k] = "I-" + raw_data[3]
                word = []
                for i in range(0, len(texts)):
                    if word == [] and texts[i][:2] == '##':
                        word.append(texts[i - 1])
                    if texts[i][:2] == '##':
                        word.append(texts[i][2:])
                    elif len(word) > 0:
                        total_word = ''.join(word)
                        if total_word in keyword:
                            piece_labels[texts.index(word[0])] = outs[ins.index(total_word)]
                            for j in range(1, len(word)):
                                piece_labels[texts.index(word[0]) + j] = 'I-' + outs[ins.index(total_word)][2:]
                        word = []
                support_data['word_piece_labels'].append(piece_labels)
            episode['support'] = support_data

            batch_data = {'seq_ins': [], 'labels': [], 'seq_outs': [], 'word_piece_marks': [], 'tokenized_texts': [],
                          'word_piece_labels': []}
            for stn in batch_examples:

                raw_data = stn

                ins = basic_tokenizer.tokenize(raw_data[0])
                batch_data['seq_ins'].append(ins)

                batch_data['labels'].append(B_label)

                outs = ['O'] * len(ins)
                keyword = basic_tokenizer.tokenize(raw_data[1])
                try:
                    outs[ins.index(keyword[0])] = "B-" + raw_data[3]
                except ValueError:
                    print('batch')
                    print(raw_data)
                    continue
                if len(keyword) > 1:
                    for k in range(1, len(keyword)):
                        outs[ins.index(keyword[0]) + k] = "I-" + raw_data[3]
                batch_data['seq_outs'].append(outs)

                texts = tokenizer.tokenize(raw_data[0])
                batch_data['tokenized_texts'].append(texts)

                piece_marks = [0] * len(texts)
                for i in range(0, len(texts)):
                    if texts[i][:2] == '##':
                        piece_marks[i] = 1
                batch_data['word_piece_marks'].append(piece_marks)

                piece_labels = ['O'] * len(piece_marks)
                if keyword[0] in texts:
                    piece_labels[texts.index(keyword[0])] = "B-" + raw_data[3]
                if len(keyword) > 1:
                    for k in range(1, len(keyword)):
                        if keyword[k] in texts:
                            piece_labels[texts.index(keyword[k])] = "I-" + raw_data[3]
                        outs[ins.index(keyword[0]) + k] = "I-" + raw_data[3]
                word = []
                for i in range(0, len(texts)):
                    if word == [] and texts[i][:2] == '##':
                        word.append(texts[i - 1])
                    if texts[i][:2] == '##':
                        word.append(texts[i][2:])
                    elif len(word) > 0:
                        total_word = ''.join(word)
                        if total_word in keyword:
                            piece_labels[texts.index(word[0])] = outs[ins.index(total_word)]
                            for j in range(1, len(word)):
                                piece_labels[texts.index(word[0]) + j] = 'I-' + outs[ins.index(total_word)][2:]
                        word = []
                batch_data['word_piece_labels'].append(piece_labels)
            episode['batch'] = batch_data
            test[B_label].append(episode)

    with open('ACL2020data/xval_ed/ed_train_'+str(order+1)+'.json', 'w') as f:
        f.write(json.dumps(test))
