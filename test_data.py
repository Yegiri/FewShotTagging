import json
domains = ['Movement', 'Conflict', 'Life', 'Sports', 'Business', 'Military', 'Music']
with open('ACL2020data/xval_snips/snips_train_2.json', 'r') as f:
    ner_data = json.loads(f.read())
print(len(ner_data['SearchScreeningEvent'][0]['batch']['word_piece_marks']))
# print()
with open('ACL2020data/xval_ed/ed_train_7.json', 'r') as f:
    data = json.loads(f.read())
print(len(data[domains[2]][0]['batch']['word_piece_marks']))
# for i in range(0, len(data[domains[1]])):
#     print(len(data[domains[1]][i]['support']['seq_ins']))

# with open('ED-data/Few-Shot_ED.json', 'r') as f:
#     ED_data = json.loads(f.read())
# B_label = []
# for i in list(ED_data.keys()):
#     if i.split('.')[0] not in B_label:
#         B_label.append(i.split('.')[0])
# print(B_label)
