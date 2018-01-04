


# with open('23828_annotated_spans.txt', 'r') as f:
#     extra_annotations = f.read().splitlines()
# with open('train_10k_annotated_spans.txt', 'r') as f:
#     train_annotations = f.read().splitlines()
# with open('23828_sentences.txt', 'r') as f:
#     extra_sentences = f.read().splitlines()
# with open('train_10k_sentences.txt', 'r') as f:
#     train_sentences = f.read().splitlines()
#
# with open('train_plus_23k_annotated_spans.txt', 'w') as f:
#     annotations = extra_annotations * 10 + train_annotations
#     f.write('\n'.join(annotations))
# print(len(set(extra_sentences)))
# with open('train_plus_23k_sentences.txt', 'w') as f:
#     sentences = extra_sentences * 10 + train_sentences
#     f.write('\n'.join(sentences))

with open('data/dev.trees', 'r') as f:
    trees = f.read().splitlines()

new_trees = []
for line in trees:
    new_tokens = []
    tokens = line.split()
    for index, token in enumerate(tokens):
        if token[0] == '(' and (not index < len(tokens) - 1 or tokens[index + 1][-1] != ')'):
            new_tokens.append('(XX')
        else:
            new_tokens.append(token)
    new_trees.append(' '.join(new_tokens))

with open('data/dev_without_labels.trees', 'w') as f:
    f.write('\n'.join(new_trees))
