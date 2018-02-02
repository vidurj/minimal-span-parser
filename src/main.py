import argparse
import itertools
import os.path
import pickle
import random
import time
from collections import namedtuple
import math
import dynet as dy
import multiprocessing
import numpy as np
from scipy import stats
import evaluate
import parse
import trees
import vocabulary
from trees import InternalParseNode, LeafParseNode, ParseNode
from collections import defaultdict
import h5py
import json
from sortedcontainers import SortedList



split_nt = namedtuple("split", ["left", "right", "oracle_split"])
label_nt = namedtuple("label", ["left", "right", "oracle_label_index"])

def check_overlap(span_a, span_b):
    return span_a[0] < span_b[0] < span_a[1] < span_b[1] or \
           span_b[0] < span_a[0] < span_b[1] < span_a[1]



def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def package(labels, file_name, append=False):
    mode = "a+" if append else "w"
    strings = []
    for label in labels:
        strings.append("{} {} {} {} {} {}".format(label["sentence_number"],
                                                  label["left"],
                                                  label["right"],
                                                  label["entropy"],
                                                  label["non_constituent_probability"],
                                                  " ".join(label["label"])))
    with open(file_name, mode) as f:
        f.write("\n" + "\n".join(strings) + "\n")


def write_seq_to_seq_data(parses_file_name, file_name):
    def to_str(sentence):
        return ' '.join([pos + ' ' + word for pos, word, in sentence])

    null_label_str = 'NULL'
    parses = load_parses(parses_file_name)
    xes = []
    yes = []
    for parse in parses:
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        subtrees = [parse]
        seen = set()
        while len(subtrees) > 0:
            tree = subtrees.pop()
            if not isinstance(tree, trees.InternalParseNode):
                assert isinstance(tree, trees.LeafParseNode), type(tree)
                continue
            seen.add((tree.left, tree.right))
            string = to_str(sentence[: tree.left]) + " [[[ " + to_str(sentence[tree.left : tree.right]) + " ]]] " + to_str(sentence[tree.right :])
            label_str = ' '.join(tree.label)
            xes.append(string)
            yes.append(label_str)

        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                if (start, end) in seen:
                    continue
                string = to_str(sentence[: start]) + " [[[ " + to_str(sentence[start: end]) + " ]]] " + to_str(sentence[end:])
                xes.append(string)
                yes.append(null_label_str)

    with open('xes-' + file_name, 'w') as f:
        f.write('\n'.join(xes))
    with open('yes-' + file_name, 'w') as f:
        f.write('\n'.join(yes))





def produce_data_for_seq_to_seq(_):
    write_seq_to_seq_data('data/train.trees', 'seq2seq-train.txt')
    write_seq_to_seq_data('data/dev.trees', 'seq2seq-dev.txt')
    write_seq_to_seq_data('data/test.trees', 'seq2seq-test.txt')


def parse_sentences(args):
    import spacy
    nlp = spacy.load('en')
    parser, _ = load_or_create_model(args, parses_for_vocab=None)
    with open(args.file_path, 'r') as f:
        sentences = f.read().splitlines()
    parses = []
    print(parser.tag_vocab.indices)
    for sentence in sentences:
        if len(parses) % 100 == 0:
            print(len(parses))
            dy.renew_cg()
        tokens = []
        for token in nlp(sentence):
            tag = token.tag_
            if tag == "-LRB-" or tag == "-RRB-":
                tag = tag[1:-1]
            if tag not in parser.tag_vocab.indices:
                print(tag, token.text)
            else:
                tokens.append((tag, token.text.replace('(', 'LRB').replace(')', 'RRB')))
        parse, _, _ = parser.span_parser(tokens, is_train=False)
        parse = parse.convert().linearize()
        parses.append(parse)
    with open('parses.txt', 'w') as f:
        f.write('\n'.join(parses))


def run_span_picking(args):
    parser, _ = load_or_create_model(args, parses_for_vocab=None)
    parses = load_parses(args.trees_path)
    # pick_spans_for_annotations(args.annotation_type,
    #                            parser,
    #                            parses,
    #                            args.expt_name,
    #                            seen=set(),
    #                            append_to_file_path=None,
    #                            fraction=1,
    #                            num_low_conf= 10 ** 8,
    #                            low_conf_cutoff=args.low_conf_cutoff,
    #                            high_conf_cutoff=args.high_conf_cutoff)


def pick_spans(label_options, size, sentence_number_to_on_spans):
    size = int(size)
    chosen_spans = []
    label_options.sort(key=lambda label: label['non_constituent_probability'])
    for label in label_options:
        must_be_off = False
        span = (label['left'], label['right'])
        sentence_number = label['sentence_number']
        if sentence_number not in sentence_number_to_on_spans:
            sentence_number_to_on_spans[sentence_number] = []
        on_spans = sentence_number_to_on_spans[sentence_number]
        for on_span in on_spans:
            if check_overlap(span, on_span):
                must_be_off = True
                break
        if not must_be_off:
            chosen_spans.append(label)
            if label['label'] != ():
                on_spans.append(span)
            if len(chosen_spans) >= size:
                break
    assert len(chosen_spans) == size, (len(chosen_spans), size)
    return chosen_spans, sentence_number_to_on_spans





def pick_spans_for_annotations(parser,
                               sentence_number_sentences_and_spans,
                               expt_name,
                               append_to_file_path,
                               num_low_conf,
                               seen,
                               low_conf_cutoff=0.05):
    if not os.path.exists(expt_name):
        os.mkdir(expt_name)
    low_confidence_labels = []
    num_processed = 0
    for sentence_number, sentence, span_to_gold_label in sentence_number_sentences_and_spans:
        if num_processed % 10000 == 0:
            print(sentence_number, len(low_confidence_labels))
        if num_processed % 100 == 0:
            dy.renew_cg()
        num_processed += 1
        _low_confidence = parser.aggressive_annotation(sentence,
                                                       sentence_number,
                                                       span_to_gold_label,
                                                       low_conf_cutoff,
                                                       seen)
        chosen_labels = []
        _low_confidence.sort(key=lambda label: - 10 * label['entropy'] + label['non_constituent_probability'])
        for label in _low_confidence:
            overlap = False
            for chosen_label in chosen_labels:
                overlap = label['left'] <= chosen_label['left'] <= label['right'] or \
                          chosen_label['left'] <= label['left'] <= chosen_label['right']
                if overlap:
                    break
            if not overlap:
                chosen_labels.append(label)

        low_confidence_labels.extend(chosen_labels)
        if sentence_number == 0:
            package(low_confidence_labels, os.path.join(expt_name, "test.txt"))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    package(low_confidence_labels, os.path.join(expt_name, timestr + "-low_confidence_labels.txt"))
    scores = np.exp(np.array([5 * annotation['entropy'] for annotation in low_confidence_labels]))
    scores = scores / np.sum(scores)
    num_samples = min(num_low_conf, len(low_confidence_labels))
    low_confidence_labels = np.random.choice(low_confidence_labels,
                                             size=num_samples,
                                             replace=False,
                                             p=scores)
    low_confidence_labels = low_confidence_labels.tolist()
    if append_to_file_path is not None:
        low_confidence_labels.sort(key=lambda label: label['sentence_number'])
        package(low_confidence_labels, append_to_file_path, append=True)


def load_training_spans(args, parser, sentence_number_sentence_and_spans):
    sentence_number_to_data = {}
    span_labels_path = os.path.join(args.expt_name, "span_labels.txt")
    if os.path.exists(span_labels_path):
        with open(span_labels_path, "r") as f:
            labels = f.read().splitlines()
            for label in labels:
                if len(label) == 0:
                    continue
                tokens = label.split()
                assert len(tokens) >= 5, "*" + label + "*"
                oracle_label = tuple(tokens[5:])
                oracle_label_index = parser.label_vocab.index(oracle_label)
                datapoint = label_nt(left=int(tokens[1]), right=int(tokens[2]),
                                     oracle_label_index=oracle_label_index)
                sentence_number = int(tokens[0])
                if sentence_number in sentence_number_to_data:
                    sentence_number_to_data[sentence_number].add(datapoint)
                else:
                    sentence_number_to_data[sentence_number] = {datapoint}
    print('loaded annotations for {} sentences'.format(len(sentence_number_to_data)))
    empty_label_index = parser.label_vocab.index(())
    for (sentence_number, sentence, _) in sentence_number_sentence_and_spans:
        if sentence_number not in sentence_number_to_data:
            continue
        seen_spans = set()
        additional_annotations = []
        for annotation in sentence_number_to_data[sentence_number]:
            start = annotation.left
            end = annotation.right
            span = (annotation.left, annotation.right)
            seen_spans.add(span)
            if annotation.oracle_label_index == empty_label_index:
                continue

            for illegal_start in range(0, start):
                for illegal_end in range(start + 1, end):
                    assert illegal_start < start < illegal_end < end
                    illegal_span = (illegal_start, illegal_end)
                    if illegal_span not in seen_spans:
                        seen_spans.add(illegal_span)
                        additional_annotations.append(
                            label_nt(left=illegal_start, right=illegal_end,
                                     oracle_label_index=empty_label_index))

            for illegal_start in range(start + 1, end):
                for illegal_end in range(end + 1, len(sentence) + 1):
                    assert start < illegal_start < end < illegal_end
                    illegal_span = (illegal_start, illegal_end)
                    if illegal_span not in seen_spans:
                        seen_spans.add(illegal_span)
                        additional_annotations.append(
                            label_nt(left=illegal_start, right=illegal_end,
                                     oracle_label_index=empty_label_index))

        sentence_number_to_data[sentence_number] = \
            sentence_number_to_data[sentence_number].union(additional_annotations)

    for k, v in sentence_number_to_data.items():
        sentence_number_to_data[k] = list(v)
    return sentence_number_to_data


def get_important_spans(parse):
    assert isinstance(parse, ParseNode)
    sentence = list(parse.leaves)
    parses = [parse]
    span_to_gold_label = {}
    constituents = []
    while len(parses) > 0:
        tree = parses.pop()
        if isinstance(tree, LeafParseNode):
            continue
        else:
            assert isinstance(tree, InternalParseNode)
        parses.extend(tree.children)
        span_to_gold_label[(tree.left, tree.right)] = tree.label
        constituents.append((tree.left, tree.right))
    for start in range(0, len(sentence) + 1):
        for end in range(start + 1, len(sentence) + 1):
            forced = False
            span = (start, end)
            if span in span_to_gold_label:
                continue
            for alt_span in constituents:
                if check_overlap(span, alt_span):
                    forced = True
                    break
            if not forced:
                span_to_gold_label[span] = ()
    return span_to_gold_label

def get_all_spans(parse):
    assert isinstance(parse, ParseNode)
    sentence = list(parse.leaves)
    parses = [parse]
    span_to_gold_label = {}
    while len(parses) > 0:
        tree = parses.pop()
        if isinstance(tree, LeafParseNode):
            continue
        else:
            assert isinstance(tree, InternalParseNode)
        parses.extend(tree.children)
        span_to_gold_label[(tree.left, tree.right)] = tree.label
    for start in range(0, len(sentence) + 1):
        for end in range(start + 1, len(sentence) + 1):
            span = (start, end)
            if span not in span_to_gold_label:
                span_to_gold_label[span] = ()
    return span_to_gold_label

def collect_random_constituents(args):
    active_learning_parses = load_parses(args.parses)
    annotations = []
    for sentence_number, parse in enumerate(active_learning_parses):
        if sentence_number % 1000 == 0:
            print(sentence_number)
        span_to_gold_label = get_all_spans(parse)
        for (left, right), label in span_to_gold_label.items():
            data_point = [str(sentence_number),
                         str(left),
                         str(right),
                         "NA",
                         "NA",
                         " ".join(label)]
            annotations.append(" ".join(data_point))
    with open(os.path.join(args.parses.split(".")[0] + "_constituents.txt"), "w") as f:
        f.write("\n".join(annotations) + "\n")


def load_or_create_model(args, parses_for_vocab):
    components = args.model_path_base.split('/')
    directory = '/'.join(components[:-1])
    if os.path.isdir(directory):
        relevant_files = [f for f in os.listdir(directory) if f.startswith(components[-1])]
    else:
        relevant_files = []
    assert len(relevant_files) <= 2, "Multiple possibilities {}".format(relevant_files)
    if len(relevant_files) > 0:
        print("Loading model from {}...".format(args.model_path_base))

        model = dy.ParameterCollection()
        [parser] = dy.load(args.model_path_base, model)
    else:
        assert parses_for_vocab is not None
        print("Constructing vocabularies using train parses...")

        tag_vocab = vocabulary.Vocabulary()
        tag_vocab.index(parse.START)
        tag_vocab.index(parse.STOP)

        word_vocab = vocabulary.Vocabulary()
        word_vocab.index(parse.START)
        word_vocab.index(parse.STOP)
        word_vocab.index(parse.UNK)

        label_vocab = vocabulary.Vocabulary()
        label_vocab.index(())

        for tree in parses_for_vocab:
            nodes = [tree]
            while nodes:
                node = nodes.pop()
                if isinstance(node, trees.InternalParseNode):
                    label_vocab.index(node.label)
                    nodes.extend(reversed(node.children))
                else:
                    assert isinstance(node, LeafParseNode)
                    tag_vocab.index(node.tag)
                    word_vocab.index(node.word)

        tag_vocab.freeze()
        word_vocab.freeze()
        label_vocab.freeze()

        def print_vocabulary(name, vocab):
            special = {parse.START, parse.STOP, parse.UNK}
            print("{} ({:,}): {}".format(
                name, vocab.size,
                sorted(value for value in vocab.values if value in special) +
                sorted(value for value in vocab.values if value not in special)))

        print("Initializing model...")
        model = dy.ParameterCollection()
        parser = parse.TopDownParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.split_hidden_dim,
            args.dropout
        )
    return parser, model


def load_parses(file_path):
    print("Loading trees from {}...".format(file_path))
    treebank = trees.load_trees(file_path)
    parses = [tree.convert() for tree in treebank]
    return parses


def check_parses(args):
    parses = load_parses(os.path.join(args.expt_name, 'active_learning.trees'))
    parser, model = load_or_create_model(args, parses)
    _, sentence_number_to_data = load_training_spans(args, parser)
    for sentence_number, data in sentence_number_to_data.items():
        gold = parses[sentence_number]
        for label in data:
            oracle_label = gold.oracle_label(label.left, label.right)
            oracle_label_index = parser.label_vocab.index(oracle_label)
            if oracle_label_index != label.oracle_label_index:
                print("*" * 60)
                print(label)
                print("*" * 60)


def print_dev_perf_by_entropy(dev_parses, matrices, span_to_entropy, parser, expt_name):
    results = []
    for sentence_number, gold in enumerate(dev_parses):
        dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in gold.leaves]
        label_probabilities = matrices[sentence_number]
        cur_index = -1
        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                cur_index += 1
                gold_label = gold.oracle_label(start, end)
                gold_label_index = parser.label_vocab.index(gold_label)
                entropy = span_to_entropy[(sentence_number, start, end)]
                predicted_label_index = np.argmax(label_probabilities[:, cur_index])
                results.append((entropy, math.log(label_probabilities[gold_label_index, cur_index] + 10 ** -8), predicted_label_index == gold_label_index))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    results.sort(key=lambda x: x[0])
    file_path = os.path.join(expt_name, timestr + '-entropy-results.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

    num_buckets = 20
    increment = int(len(results) / num_buckets)
    output_str = "\n"
    for i in range(num_buckets):
        temp = results[:increment]
        entropies, log_probabilities, accuracies = zip(*temp)
        output_str += str(np.mean(entropies)) + ' ' + str(np.mean(log_probabilities)) + ' ' + str(np.mean(accuracies)) + '\n'
        results = results[increment:]
    file_path = os.path.join(expt_name, 'dev_entropy.txt')
    with open(file_path, 'a+') as f:
        f.write(output_str)



def run_training_on_spans(args):
    assert args.batch_size == 100, args.batch_size
    return_code = os.system('cp -r src {}/'.format(args.expt_name))
    assert return_code == 0
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)



    all_parses = load_parses(args.train_path)

    with open(os.path.join(args.expt_name, "train_tree_indices.txt"), 'r') as f:
        train_tree_indices = [int(x) for x in f.read().strip().splitlines()]
    train_tree_indices_set = set(train_tree_indices)
    print('training on', len(train_tree_indices), 'full trees')

    parser, model = load_or_create_model(args, all_parses)


    train_sentence_number_to_annotations = {}
    active_learning_sentences_and_spans = []
    for sentence_number in range(len(all_parses)):
        parse = all_parses[sentence_number]
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        span_to_gold_label = get_all_spans(parse)
        assert span_to_gold_label[(0, len(sentence))] != ()
        parse = parse.clean_up_punctuation()
        all_parses[sentence_number] = parse
        span_to_gold_label = get_all_spans(parse)
        assert span_to_gold_label[(0, len(sentence))] != ()
        if sentence_number in train_tree_indices_set:
            data = []
            for (left, right), oracle_label in list(span_to_gold_label.items()):
                if (left, right) != (0, len(sentence)) and (sentence[left][0] in trees.deletable_tags or sentence[right - 1][0] in trees.deletable_tags):
                    assert oracle_label == parser.empty_label, (oracle_label, left, right, 0, len(sentence), sentence[left: right], sentence)
                else:
                    oracle_label_index = parser.label_vocab.index(oracle_label)
                    data.append(label_nt(left=left, right=right, oracle_label_index=oracle_label_index))
            train_sentence_number_to_annotations[sentence_number] = data
        else:
            active_learning_sentences_and_spans.append((sentence_number, sentence, span_to_gold_label))


    print("Loaded {:,} training examples.".format(len(train_tree_indices)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")

    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    span_to_entropy = {}

    dev_parses = [x.convert() for x in dev_treebank]

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        # total_loglikelihood = 0
        matrices = []
        fill_span_to_entropy = len(span_to_entropy) == 0
        for sentence_number, tree in enumerate(dev_treebank):
            if sentence_number % 100 == 0:
                dy.renew_cg()
                cur_word_index = 0
                batch_number = int(sentence_number / 100)
                h5f = h5py.File('ptb_elmo_embeddings/dev/batch_{}_embeddings.h5'.format(batch_number), 'r')
                embedding_array = h5f['embeddings'][:, :, :]
                elmo_embeddings = dy.inputTensor(embedding_array)
                h5f.close()

            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            predicted, _, label_probabilities = parser.span_parser(sentence, is_train=False, elmo_embeddings=elmo_embeddings, cur_word_index=cur_word_index)
            cur_word_index += len(sentence)

            # total_loglikelihood += log_likelihood
            dev_predicted.append(predicted.convert())
            matrices.append(label_probabilities)
            cur_index = -1
            if fill_span_to_entropy:
                for start in range(0, len(sentence)):
                    for end in range(start + 1, len(sentence) + 1):
                        cur_index += 1
                        entropy = stats.entropy(label_probabilities[:, cur_index])
                        span_to_entropy[(sentence_number, start, end)] = entropy
                assert cur_index == label_probabilities.shape[1] - 1, (cur_index, label_probabilities.shape)


        print_dev_perf_by_entropy(dev_parses, matrices, span_to_entropy, parser, args.expt_name)

        if args.erase_labels:
            dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted,
                                                       args=args,
                                                       erase_labels=True,
                                                       name="dev-without-labels")
            print("dev-fscore without labels", dev_fscore_without_labels)

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted, args=args,
                                    name="dev-regular")

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])
            return True, dev_fscore
        else:
            latest_model_path = "{}_latest_model".format(args.model_path_base, dev_fscore.fscore)
            for ext in [".data", ".meta"]:
                path = latest_model_path + ext
                if os.path.exists(path):
                    print("Removing previous model file {}...".format(path))
                    os.remove(path)

            print("Saving new model to {}...".format(latest_model_path))
            dy.save(latest_model_path, [parser])
            return False, dev_fscore

    num_batches = 0
    total_batch_loss = prev_total_batch_loss = None


    annotated_sentence_number_to_annotations = load_training_spans(args, parser, active_learning_sentences_and_spans)
    train_sentence_number_to_annotations.update(annotated_sentence_number_to_annotations)
    all_sentence_number_to_annotations = train_sentence_number_to_annotations
    seen = set()
    for sentence_number, annotations in annotated_sentence_number_to_annotations.items():
        for label in annotations:
            span = (label.left, label.right)
            seen.add((span, sentence_number))
    return_code = os.system('echo "test"')
    assert return_code == 0
    batch_numbers = list(range(398))
    num_trees = len(all_sentence_number_to_annotations)
    print('num trees', num_trees)
    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        if epoch > 1:
            is_best, dev_score = check_dev()
        else:
            is_best = True
            dev_score = None

        if dev_score is not None:
            perf_summary = '\n' + '-' * 40 + '\n' + str(dev_score) + '\n'
            with open("performance.txt", "a+") as f:
                f.write(perf_summary)
            return_code = os.system("date >> performance.txt")
            assert return_code == 0
            span_labels_path = os.path.join(args.expt_name, "span_labels.txt")
            if os.path.exists(span_labels_path):
                return_code = os.system(
                    "wc -l {} >> performance.txt".format(span_labels_path))
                assert return_code == 0

        print("Total batch loss", total_batch_loss, "Prev batch loss", prev_total_batch_loss)
        if args.annotation_type != "none" and not is_best:
            print("Adding more training data")
            pick_spans_for_annotations(parser,
                                       active_learning_sentences_and_spans,
                                       args.expt_name,
                                       os.path.join(args.expt_name, "span_labels.txt"),
                                       seen=seen,
                                       num_low_conf=args.num_low_conf,
                                       low_conf_cutoff=float(args.low_conf_cutoff))
            annotated_sentence_number_to_annotations = load_training_spans(args, parser, active_learning_sentences_and_spans)
            train_sentence_number_to_annotations.update(annotated_sentence_number_to_annotations)
            all_sentence_number_to_annotations = train_sentence_number_to_annotations
            for sentence_number, annotations in annotated_sentence_number_to_annotations.items():
                for label in annotations:
                    span = (label.left, label.right)
                    seen.add((span, sentence_number))
            prev_total_batch_loss = None
        else:
            prev_total_batch_loss = total_batch_loss
        for _ in range(3):
            np.random.shuffle(batch_numbers)
            epoch_start_time = time.time()
            annotation_index = 0
            num_batches = 0
            total_batch_loss = 0
            for batch_number in batch_numbers:
                dy.renew_cg()
                batch_losses = []
                elmo_embeddings = None
                cur_word_index = 0

                for offset in range(args.batch_size):
                    sentence_number = batch_number * 100 + offset
                    tree = all_parses[sentence_number]
                    if sentence_number not in all_sentence_number_to_annotations:
                        cur_word_index += len(tree.leaves)
                        continue
                    elif elmo_embeddings is None:
                        h5f = h5py.File('ptb_elmo_embeddings/train/batch_{}_embeddings.h5'.format(
                            batch_number), 'r')
                        elmo_embeddings = dy.inputTensor(h5f['embeddings'][:, :, :])
                        h5f.close()

                    sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
                    annotation_index += 1
                    loss = parser.train_on_partial_annotation(
                            sentence,
                            all_sentence_number_to_annotations[sentence_number],
                            elmo_vecs=elmo_embeddings,
                            cur_word_index=cur_word_index
                    )
                    batch_losses.append(loss)
                    total_processed += 1
                    current_processed += 1
                    cur_word_index += len(tree.leaves)

                if len(batch_losses) > 0:
                    print("elmo weights", parser.elmo_weights.as_array())
                    batch_loss = dy.average(batch_losses)
                    batch_loss.backward()
                    trainer.update()
                    batch_loss_value = batch_loss.scalar_value()
                    total_batch_loss += batch_loss_value
                    num_batches += 1

                    print(
                        "epoch {:,} "
                        "batch {:,}/{:,} "
                        "processed {:,} "
                        "batch-loss {:.4f} "
                        "epoch-elapsed {} "
                        "total-elapsed {}".format(
                            epoch,
                            num_batches,
                            len(batch_numbers),
                            total_processed,
                            batch_loss_value,
                            format_elapsed(epoch_start_time),
                            format_elapsed(start_time),
                        )
                    )


def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_parse = load_parses(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_parse)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    parser, model = load_or_create_model(args, train_parse)
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()


    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            predicted, _, _ = parser.span_parser(sentence, is_train=False)
            dev_predicted.append(predicted.convert())

        if args.erase_labels:
            dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted,
                                                       args=args,
                                                       erase_labels=True,
                                                       name="dev-without-labels")
            print("dev-fscore without labels", dev_fscore_without_labels)

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted, args=args,
                                    name="dev")

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                if args.train_on_subtrees and random.random() < 0.5:
                    subtrees = [tree]
                    relevant_trees = [tree]
                    while len(relevant_trees) > 0:
                        tree = relevant_trees.pop()
                        if not isinstance(tree, trees.InternalParseNode):
                            assert isinstance(tree, trees.LeafParseNode)
                            continue
                        if len(tree.leaves) > 5:
                            continue
                        relevant_trees.extend(tree.children)
                        subtrees.append(tree)
                    tree = random.choice(subtrees)
                    tree.reset(0)
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
                _, loss = parser.span_parser(sentence, is_train=True, gold=tree)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // (args.batch_size + 1),
                    int(np.ceil(len(train_parse) / (args.batch_size + 1))),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()




def run_test_qbank(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    test_path = 'questionbank/qbank.{}.trees'.format(args.split)
    test_treebank = trees.load_trees(test_path)
    test_embeddings = []
    if args.split == 'train':
        indices = range(2000)
    elif args.split == 'dev':
        indices = range(2000, 3000)
    else:
        assert args.split == 'test', args.split
        indices = range(3000, 4000)

    with h5py.File('question_bank_elmo_embeddings.hdf5', 'r') as h5f:
        for index in indices:
            test_embeddings.append(h5f[str(index)][:, :, :])

    test_embeddings_np = np.swapaxes(np.concatenate(test_embeddings, axis=1), axis1=0, axis2=1)

    dev_predicted = []
    cur_word_index = 0
    for dev_index, tree in enumerate(test_treebank):
        if dev_index % 100 == 0:
            dy.renew_cg()
            test_embeddings = dy.inputTensor(test_embeddings_np)
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        predicted, _, _ = parser.span_parser(sentence, is_train=False,
                                             elmo_embeddings=test_embeddings,
                                             cur_word_index=cur_word_index)
        dev_predicted.append(predicted.convert())
        cur_word_index += len(sentence)

    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, dev_predicted,
                                               args=args,
                                               erase_labels=True,
                                               name="without-labels")
    print("dev-fscore without labels", dev_fscore_without_labels)

    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, dev_predicted,
                                               args=args,
                                               erase_labels=True,
                                               flatten=True,
                                               name="without-label-flattened")
    print("dev-fscore without labels and flattened", dev_fscore_without_labels)

    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, dev_predicted,
                                               args=args,
                                               erase_labels=False,
                                               flatten=True,
                                               name="flattened")
    print("dev-fscore with labels and flattened", dev_fscore_without_labels)

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, dev_predicted, args=args,
                                 name="regular")

    print("regular", test_fscore)

def run_train_question_bank(args):
    train_path = 'questionbank/qbank.train.trees'
    dev_path = 'questionbank/qbank.dev.trees'
    print("Loading training trees from {}...".format(train_path))
    train_parse = load_parses(train_path)
    print("Loaded {:,} training examples.".format(len(train_parse)))

    print("Loading development trees from {}...".format(dev_path))
    dev_treebank = trees.load_trees(dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    wsj_train = load_parses('data/train.trees')

    parser, model = load_or_create_model(args, train_parse + wsj_train)
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse)
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    train_embeddings = []
    dev_embeddings = []
    with h5py.File('question_bank_elmo_embeddings.hdf5', 'r') as h5f:
        for index in range(len(train_parse)):
            train_embeddings.append(h5f[str(index)][:, :, :])
        for index in range(len(train_parse), len(train_parse) + len(dev_treebank)):
            dev_embeddings.append(h5f[str(index)][:, :, :])

    train_embeddings_np = np.swapaxes(np.concatenate(train_embeddings, axis=1), axis1=0, axis2=1)
    dev_embeddings_np = np.swapaxes(np.concatenate(dev_embeddings, axis=1), axis1=0, axis2=1)


    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        cur_word_index = 0
        for dev_index, tree in enumerate(dev_treebank):
            if dev_index % 100 == 0:
                dy.renew_cg()
                dev_embeddings = dy.inputTensor(dev_embeddings_np)
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            predicted, _, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=dev_embeddings, cur_word_index=cur_word_index)
            dev_predicted.append(predicted.convert())
            cur_word_index += len(sentence)

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted, args=args,
                                    name="dev")

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])


    sentence_number_to_sentence_and_word_index = {}

    word_index = 0
    for index, tree in enumerate(train_parse):
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        sentence_number_to_sentence_and_word_index[index] = (sentence, word_index)
        word_index += len(sentence)


    for epoch in itertools.count(start=1):

        tree_indices = list(range(len(train_parse)))
        np.random.shuffle(tree_indices)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()
            train_embeddings = dy.inputTensor(train_embeddings_np)
            batch_losses = []
            for index in tree_indices[start_index:start_index + args.batch_size]:
                tree = train_parse[index]
                sentence, cur_word_index = sentence_number_to_sentence_and_word_index[index]
                _, loss = parser.span_parser(sentence, is_train=True, gold=tree, elmo_embeddings=train_embeddings, cur_word_index=cur_word_index)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // (args.batch_size + 1),
                    int(np.ceil(len(train_parse) / (args.batch_size + 1))),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()


def run_train_question_bank_and_wsj(args):
    train_path = 'questionbank/qbank.train.trees'
    dev_path = 'questionbank/qbank.dev.trees'
    print("Loading training trees from {}...".format(train_path))
    train_parse = load_parses(train_path)
    print("Loaded {:,} training examples.".format(len(train_parse)))

    print("Loading development trees from {}...".format(dev_path))
    dev_treebank = trees.load_trees(dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    wsj_train = load_parses('data/train.trees')

    parser, model = load_or_create_model(args, train_parse + wsj_train)
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse)
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    train_embeddings = []
    dev_embeddings = []
    with h5py.File('question_bank_elmo_embeddings.hdf5', 'r') as h5f:
        for index in range(len(train_parse)):
            train_embeddings.append(h5f[str(index)][:, :, :])
        for index in range(len(train_parse), len(train_parse) + len(dev_treebank)):
            dev_embeddings.append(h5f[str(index)][:, :, :])

    train_embeddings_np = np.swapaxes(np.concatenate(train_embeddings, axis=1), axis1=0, axis2=1)
    dev_embeddings_np = np.swapaxes(np.concatenate(dev_embeddings, axis=1), axis1=0, axis2=1)


    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        cur_word_index = 0
        for dev_index, tree in enumerate(dev_treebank):
            if dev_index % 100 == 0:
                dy.renew_cg()
                dev_embeddings = dy.inputTensor(dev_embeddings_np)
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            predicted, _, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=dev_embeddings, cur_word_index=cur_word_index)
            dev_predicted.append(predicted.convert())
            cur_word_index += len(sentence)

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted, args=args,
                                    name="dev")

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])


    sentence_number_to_sentence_and_word_index = {}

    word_index = 0
    for index, tree in enumerate(train_parse):
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        sentence_number_to_sentence_and_word_index[index] = (sentence, word_index)
        word_index += len(sentence)


    for epoch in itertools.count(start=1):

        tree_indices = list(range(len(train_parse)))
        np.random.shuffle(tree_indices)
        epoch_start_time = time.time()
        wsj_indices = []
        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()
            train_embeddings = dy.inputTensor(train_embeddings_np)
            batch_losses = []
            for index in tree_indices[start_index:start_index + args.batch_size]:
                tree = train_parse[index]
                sentence, cur_word_index = sentence_number_to_sentence_and_word_index[index]
                _, loss = parser.span_parser(sentence, is_train=True, gold=tree, elmo_embeddings=train_embeddings, cur_word_index=cur_word_index)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            if not wsj_indices:
                wsj_indices = list(range(398))
                random.shuffle(wsj_indices)
            wsj_batch_index = wsj_indices.pop()
            h5f = h5py.File(
                'ptb_elmo_embeddings/train/batch_{}_embeddings.h5'.format(wsj_batch_index), 'r')
            wsj_train_embeddings = dy.inputTensor(h5f['embeddings'][:, :, :])
            h5f.close()
            wsj_cur_word_index = 0
            for index in range(wsj_batch_index * 100, wsj_batch_index * 100 + 100):
                tree = wsj_train[index]
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
                _, loss = parser.span_parser(sentence, is_train=True, gold=tree,
                                             elmo_embeddings=wsj_train_embeddings,
                                             cur_word_index=wsj_cur_word_index)
                wsj_cur_word_index += len(sentence)
                batch_losses.append(loss)

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // (args.batch_size + 1),
                    int(np.ceil(len(train_parse) / (args.batch_size + 1))),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()


def fine_tune_confidence(args):
    file_name = 'dev'
    parses = load_parses('data/{}.trees'.format(file_name))
    print("Loaded {:,} test examples.".format(len(parses)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    lmbd_param_collection = model.add_subcollection("Lambda")
    [parser] = dy.load(args.model_path_base, model)
    lmbd_parameter = lmbd_param_collection.add_parameters((1, 1), dy.ConstInitializer(998))
    #998.97674561]] [ 0.00465745  1.75719917 -1.226492
    alpha_parameter = lmbd_param_collection.parameters_from_numpy(np.array([0.00465745,  1.75719917, -1.226492]))
    trainer = dy.AdamTrainer(lmbd_param_collection)
    print(trainer.learning_rate)
    trainer.restart(learning_rate=0.03)
    print(trainer.learning_rate)
    print("Parsing test sentences...")
    while True:
        cur_word_index = None
        for index, tree in enumerate(parses):
            if index % 100 == 0:
                dy.renew_cg()
                lmbd = dy.parameter(lmbd_parameter)
                alpha = dy.parameter(alpha_parameter)
                batch_losses = []
                if cur_word_index is not None:
                    assert cur_word_index == len(embedding_array), (cur_word_index, len(embedding_array))
                cur_word_index = 0
                batch_number = int(index / 100)
                embedding_file_name = 'ptb_elmo_embeddings/{}/batch_{}_embeddings.h5'.format(file_name, batch_number)
                h5f = h5py.File(embedding_file_name, 'r')
                embedding_array = h5f['embeddings'][:, :, :]
                elmo_embeddings = dy.inputTensor(embedding_array)
                h5f.close()
                print(index)
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            loss = parser.fine_tune_confidence(sentence, lmbd, alpha, elmo_embeddings,
                                                      cur_word_index, tree)
            cur_word_index += len(sentence)
            batch_losses.append(loss)

            if index % 100 == 99:
                batch_loss = dy.average(batch_losses)
                batch_loss_value = batch_loss.scalar_value()
                batch_loss.backward()
                trainer.update()
                print(batch_loss_value, lmbd_parameter.as_array(), alpha_parameter.as_array())
                print('-' * 40)


def kbest(args):
    empty_label_index = 0
    labels = [(), ('XX',)]
    sentence, num_trees, label_log_probabilities_np, span_to_index = args
    correction_term = np.sum(label_log_probabilities_np[0, :])
    label_log_probabilities_np -= label_log_probabilities_np[0, :]

    cache = {}

    # tree_to_string = {}
    #
    # def compute_string(tree):
    #     if tree not in tree_to_string:
    #         deleted = tree.delete_punctuation()
    #         if deleted is not None:
    #             tree_string = deleted.convert().linearize()
    #         else:
    #             tree_string = ""
    #         tree_to_string[tree] = tree_string
    #     return tree_to_string[tree]

    def helper(left, right):
        assert left < right
        key = (left, right)
        if key in cache:
            return cache[key]

        span_index = span_to_index[(left, right)]
        actions = list(enumerate(label_log_probabilities_np[:, span_index]))
        if left == 0 and right == len(sentence):
            actions = actions[1:]
        actions.sort(key=lambda x: - x[1])
        actions = actions[:num_trees]

        if right - left == 1:
            tag, word = sentence[left]
            leaf = LeafParseNode(left, tag, word)
            options = []
            for label_index, score in actions:
                if label_index != empty_label_index:
                    label = labels[label_index]
                    tree = InternalParseNode(label, [leaf])
                else:
                    tree = leaf
                options.append(([tree], score))
            cache[key] = options
        else:
            children_options = SortedList(key=lambda x: - x[1])
            for split in range(left + 1, right):
                left_trees_options = helper(left, split)
                right_trees_options = helper(split, right)
                for (left_trees, left_score) in left_trees_options:
                    if len(left_trees) > 1:
                        # To avoid duplicates, we require that left trees are constituents
                        continue
                    for (right_trees, right_score) in right_trees_options:
                        children = left_trees + right_trees
                        score = left_score + right_score
                        if len(children_options) < num_trees:
                            children_options.add((children, score))
                        elif children_options[-1][1] < score:
                            del children_options[-1]
                            children_options.add((children, score))

            options = SortedList(key=lambda x: - x[1])
            string_to_score = {}
            for (label_index, action_score) in actions:
                for (children, children_score) in children_options:
                    option_score = action_score + children_score
                    if label_index != 0:
                        label = labels[label_index]
                        tree = InternalParseNode(label, children)
                        option = [tree]
                    else:
                        option = children
                    option_string = ''.join([compute_string(tree) for tree in option])
                    if option_string in string_to_score and option_score <= string_to_score[option_string]:
                        continue

                    string_to_score[option_string] = option_score
                    if len(options) < num_trees:
                        options.add((option, option_score))
                    elif options[-1][1] < option_score:
                        del options[-1]
                        options.add((option, option_score))
                    else:
                        break
            cache[key] = options
        return cache[key]

    trees_and_scores = helper(0, len(sentence))[:num_trees]
    trees = []
    scores = []
    for tree, score in trees_and_scores:
        assert len(tree) == 1
        trees.append(tree[0].convert())
        scores.append(score + correction_term)
    return trees, scores


tree_to_string = {}
def compute_string(tree):
    if tree not in tree_to_string:
        deleted = tree.delete_punctuation()
        if deleted is not None:
            tree_string = deleted.convert().linearize()
        else:
            tree_string = ""
        tree_to_string[tree] = tree_string
    return tree_to_string[tree]

def compute_kbest_f1(args):
    sizes = [1, 5, 10, 15, 20, 50, 100, 200, 300, 400, 500]
    sizes.sort()
    sizes = [x for x in sizes if x <= args.num_trees]
    test_parses = load_parses(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_parses)))
    num_excess_trees = len(test_parses) % 100
    if num_excess_trees != 0:
        print('last {} parses are skipped by current elmo vecs'.format(num_excess_trees))
        test_parses = test_parses[:-num_excess_trees]

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    file_name = args.test_path.split('/')[-1]
    assert file_name.endswith('.trees'), args.test_path
    file_name = file_name[:-6]
    assert file_name == 'test' or file_name == 'dev' or file_name == 'train', args.test_path

    print("Parsing test sentences...")
    data = []
    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool()
    for index, tree in enumerate(test_parses):
        if index % 100 == 0:
            dy.renew_cg()
            cur_word_index = 0
            batch_number = int(index / 100)
            embedding_file_name = 'ptb_elmo_embeddings/{}/batch_{}_embeddings.h5'.format(file_name,
                                                                                         batch_number)
            h5f = h5py.File(embedding_file_name, 'r')
            embedding_array = h5f['embeddings'][:, :, :]
            elmo_embeddings = dy.inputTensor(embedding_array)
            h5f.close()
            print(index)
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        (label_log_probabilities_np, span_to_index) = parser.get_distribution_for_kbest(sentence, elmo_embeddings, cur_word_index)
        cur_word_index += len(sentence)
        data.append((sentence, args.num_trees, label_log_probabilities_np, span_to_index))

    responses = pool.map(kbest, data, chunksize=int(math.floor(len(data) / num_cpus)))
    all_scores = []
    predicted_acc = []
    gold_acc = []
    num_trees = []
    for (tree, (predicted_trees, scores)) in zip(test_parses, responses):
        # assert len(predicted_trees_and_scores) == args.num_trees, (sentence, len(predicted_trees_and_scores))
        # predicted, additional_info, _ = parser.span_parser(sentence, is_train=False, gold=tree)
        gold_treebank_tree = tree.convert()
        predicted_acc.extend(predicted_trees)
        gold_acc.extend([gold_treebank_tree for _ in predicted_trees])
        all_scores.extend(scores)
        assert len(predicted_trees) == len(scores)
        num_trees.append(len(predicted_trees))

    evaluate.evalb(args.evalb_dir, gold_acc, predicted_acc, args=args,
                                 name="bestk", flatten=True, erase_labels=True)
    file_path = os.path.join(args.expt_name, 'bestk-output.txt')
    with open(file_path, 'r') as f:
        text = f.read().strip().splitlines()
    flag = False
    size_to_scores = {}
    for size in sizes:
        size_to_scores[size] = []

    thresholds = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    thresholds.sort()
    threshold_to_scores = {}
    for threshold in thresholds:
        threshold_to_scores[threshold] = []
    ks_to_ks = {}
    for ks in sizes:
        ks_to_ks[ks] = []
    threshold_to_ks = {}
    for threshold in threshold_to_scores.keys():
        threshold_to_ks[threshold] = []
    buffer = []
    buffer_probability = 0
    cur_index = 0
    cur_tree_number = 0
    for line in text:
        if line == '============================================================================':
            if flag:
                assert buffer == [], len(buffer)
                break
            flag = True
        elif flag:
            tokens = line.split()
            matched_brackets, gold_brackets, predicted_brackets = [int(x) for x in tokens[5:8]]
            precision = matched_brackets / predicted_brackets
            recall = matched_brackets / gold_brackets
            if matched_brackets == 0:
                f1 = 0
            else:
                f1 = 2 / (1 / precision + 1 / recall)
            buffer.append((matched_brackets, gold_brackets, predicted_brackets, f1))
            print(all_scores[cur_index])
            probability = math.exp(all_scores[cur_index])
            cur_index += 1
            old_buffer_probability = buffer_probability
            buffer_probability += probability

            best_score = max(buffer, key=lambda x: x[3])

            if len(buffer) in size_to_scores:
                size_to_scores[len(buffer)].append(best_score)

            for threshold in thresholds:
                if old_buffer_probability < threshold <= buffer_probability:
                    threshold_to_scores[threshold].append(best_score)
                    threshold_to_ks[threshold].append(len(buffer))

            # Must be num_trees since we are generating num_trees and we will be out of alignment
            if len(buffer) == num_trees[cur_tree_number]:
                for threshold in thresholds:
                    if buffer_probability < threshold:
                        print(threshold, 'was too high for', args.num_trees, 'trees')
                        threshold_to_scores[threshold].append(best_score)
                        threshold_to_ks[threshold].append(len(buffer))
                buffer = []
                buffer_probability = 0
                cur_tree_number += 1

    def print_stats(best_scores, name):
        total_matched_brackets = np.sum([x[0] for x in best_scores])
        total_gold_brackets = np.sum([x[1] for x in best_scores])
        total_predicted_brackets = np.sum([x[2] for x in best_scores])
        complete_match_fraction = np.mean([x == y == z for x, y, z, _ in best_scores])
        recall = total_matched_brackets / total_gold_brackets
        precision = total_matched_brackets / total_predicted_brackets
        f1 = 2 / (1 / precision + 1 / recall)
        print('precision:', precision,
              'recall:', recall,
              'f1:', f1,
              'complete match fraction:', complete_match_fraction,
              'num sentences:')
        with open(name + '.pickle', 'wb') as f:
            pickle.dump(best_scores, f)


    for size in sizes:
        print(size)
        if len(size_to_scores[size]) != len(test_parses):
            print(size, 'has only', len(size_to_scores[size]), 'instances instead of', len(test_parses))
        print_stats(size_to_scores[size], 'k_equals_' + str(size))
        print('-' * 50)

    for probability in threshold_to_scores.keys():
        print('threshold', probability)
        assert len(threshold_to_ks[probability]) == len(test_parses)
        assert len(threshold_to_scores[probability]) == len(test_parses)
        print('avg k', np.mean(threshold_to_ks[probability]))
        print_stats(threshold_to_scores[probability], 'probability_mass_' + str(probability))
        print('-' * 50)



def produce_parse_forests(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    test_parse = [tree.convert() for tree in test_treebank]
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    test_predicted = []
    num_trees_per_parse = []
    forest_prob_masses = []
    for tree in test_parse:
        if len(test_predicted) % 100 == 0:
            print(len(test_predicted))
            dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        forest, forest_prob_mass = parser.produce_parse_forest(sentence, required_probability_mass=0.9)
        test_predicted.append(forest)
        num_trees = np.product([len(options) for options in forest.values()])
        num_trees_per_parse.append(num_trees)
        forest_prob_masses.append(forest_prob_mass)
    print(num_trees_per_parse)
    print("avg", np.mean(num_trees_per_parse), "median", np.median(num_trees_per_parse))
    print(forest_prob_masses)
    print("avg", np.mean(forest_prob_masses), "median", np.median(forest_prob_masses))


def run_test(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)
    print("Loading test trees from {}...".format(args.test_path))
    file_name = args.test_path.split('/')[-1]
    assert file_name.endswith('.trees'), args.test_path
    file_name = file_name[:-6]
    assert file_name == 'test' or file_name == 'dev' or file_name == 'train', args.test_path
    test_treebank = trees.load_trees(args.test_path)
    num_excess_trees = len(test_treebank) % 100
    if num_excess_trees != 0:
        print('last {} parses are skipped by current elmo vecs'.format(num_excess_trees))
        test_treebank = test_treebank[:-num_excess_trees]

    test_parse = [tree.convert() for tree in test_treebank]
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    test_predicted = []
    start_time = time.time()
    total_log_likelihood = 0
    total_confusion_matrix = {}
    total_turned_off = 0
    ranks = []
    print('elmo weights', parser.elmo_weights.as_array())
    for tree_index, tree in enumerate(test_parse):
        if tree_index % 100 == 0:
            dy.renew_cg()
            cur_word_index = 0
            batch_number = int(tree_index / 100)
            embedding_file_name = 'ptb_elmo_embeddings/{}/batch_{}_embeddings.h5'.format(file_name, batch_number)
            h5f = h5py.File(embedding_file_name, 'r')
            embedding_array = h5f['embeddings'][:, :, :]
            elmo_embeddings = dy.inputTensor(embedding_array)
            h5f.close()
            print(tree_index)
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        predicted, additional_info, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=elmo_embeddings, cur_word_index=cur_word_index, gold=tree)
        cur_word_index += len(sentence)
        rank = additional_info[3]
        ranks.append(rank)
        # total_log_likelihood += _log_likelihood
        test_predicted.append(predicted.convert())
        # total_turned_off += _turned_off
        # for k, v in _confusion_matrix.items():
        #     if k in total_confusion_matrix:
        #         total_confusion_matrix[k] += v
        #     else:
        #         total_confusion_matrix[k] = v
    print("total time", time.time() - start_time)
    print("total loglikelihood", total_log_likelihood)
    print("total turned off", total_turned_off)
    print(total_confusion_matrix)

    print(ranks)
    print("avg", np.mean(ranks), "median", np.median(ranks))

    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted,
                                               args=args,
                                               erase_labels=True,
                                               name="without-labels")
    print("dev-fscore without labels", dev_fscore_without_labels)

    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted,
                                               args=args,
                                               erase_labels=True,
                                               flatten=True,
                                               name="without-label-flattened")
    print("dev-fscore without labels and flattened", dev_fscore_without_labels)

    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted,
                                               args=args,
                                               erase_labels=False,
                                               flatten=True,
                                               name="flattened")
    print("dev-fscore with labels and flattened", dev_fscore_without_labels)

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted, args=args,
                                 name="regular")

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )
    with open(os.path.join(args.expt_name, "confusion_matrix.pickle"), "wb") as f:
        pickle.dump(total_confusion_matrix, f)


def collect_mistakes(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)
    data_path = 'data/dev.trees'
    print("Loading test trees from {}...".format(data_path))
    parses = load_parses(data_path)


    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)
    print(parser.label_vocab.counts)

    print("Parsing test sentences...")
    errors = []
    for tree_index, tree in enumerate(parses):
        if tree_index % 100 == 0:
            dy.renew_cg()
            cur_word_index = 0
            batch_number = int(tree_index / 100)
            embedding_file_name = 'ptb_elmo_embeddings/dev/batch_{}_embeddings.h5'.format(batch_number)
            h5f = h5py.File(embedding_file_name, 'r')
            embedding_array = h5f['embeddings'][:, :, :]
            elmo_embeddings = dy.inputTensor(embedding_array)
            h5f.close()
            print(tree_index)
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        words = [word for pos, word in sentence]
        span_to_index, label_log_probabilities = parser.compute_label_distributions(sentence,
                                                                                  is_train=True,
                                                                                  elmo_embeddings=elmo_embeddings,
                                                                                  cur_word_index=cur_word_index)
        label_log_probabilities = label_log_probabilities.npvalue()
        cur_word_index += len(sentence)
        span_to_label = get_all_spans(tree)


        for span, label in span_to_label.items():
            start = span[0]
            end = span[1] - 1
            # while start < len(sentence) and sentence[start][0] in deletable_tags:
            #     start += 1
            # while end >= 0 and sentence[end][0] in deletable_tags:
            #     end -= 1
            if end <= start:
                start = span[0]
                end = span[1] - 1
            assert end < len(sentence), (end, len(sentence))
            label_index = parser.label_vocab.index(label)
            span_index = span_to_index[span]
            nc_prob = math.exp(label_log_probabilities[parser.empty_label_index, span_index])
            if label_index == parser.empty_label_index:
                prob = nc_prob
            else:
                prob = 1 - nc_prob
            if prob < 0.5:
                if label == ():
                    label_str = 'NC'
                else:
                    label_str = ' '.join(label)


                string = ' '.join(words[:span[0]]) + ' [[[ ' + ' '.join(words[span[0]:span[1]]) + ' ]]] ' + ' '.join(words[span[1]:])
                errors.append((prob, label_str, string.strip()))

    errors.sort(key=lambda x: x[0])
    error_string = ""
    for prob, label, string in errors:
        error_string += str(prob) + '\t' + label + '\t' + string + '\n'
    with open(args.expt_name + '/errors.txt', 'w') as f:
        f.write(error_string)



def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train-wsj-qb")
    subparser.set_defaults(callback=run_train_question_bank_and_wsj)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--batch-size", type=int, default=50)
    subparser.add_argument("--expt-name", required=True)


    subparser = subparsers.add_parser("train-question-bank")
    subparser.set_defaults(callback=run_train_question_bank)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--batch-size", type=int, default=100)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--explore", action="store_true")
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", required=True)
    subparser.add_argument("--dev-path", required=True)
    subparser.add_argument("--sentences", default="NONE")
    subparser.add_argument("--annotations", default="NONE")
    subparser.add_argument("--batch-size", type=int, default=10)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--separate-left-right", action="store_true")
    subparser.add_argument("--erase-labels", action="store_false", default=True)
    subparser.add_argument("--train-on-subtrees", action="store_true", default=False)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--make-trees", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("test-qb")
    subparser.set_defaults(callback=run_test_qbank)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--split", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("collect-errors")
    subparser.set_defaults(callback=collect_mistakes)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("parse-forest")
    subparser.set_defaults(callback=produce_parse_forests)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--test-path", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("seq2seq-data")
    subparser.set_defaults(callback=produce_data_for_seq_to_seq)

    subparser = subparsers.add_parser("bestk-test")
    subparser.set_defaults(callback=compute_kbest_f1)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--test-path", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--num-trees", required=True, type=int)

    subparser = subparsers.add_parser("pick-spans")
    subparser.set_defaults(callback=run_span_picking)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--trees-path", required=True)
    subparser.add_argument("--annotation-type",
                           choices=["random-spans", "incorrect-spans", "uncertainty"],
                           required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--low-conf-cutoff", default=0.005, type=float)
    subparser.add_argument("--high-conf-cutoff", default=0.0001, type=float)

    subparser = subparsers.add_parser("check-labels")
    subparser.set_defaults(callback=check_parses)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("active-learning")
    subparser.set_defaults(callback=run_training_on_spans)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--annotation-type",
                           choices=["random-spans", "incorrect-spans", "uncertainty", "none"],
                           required=True)
    subparser.add_argument("--num-low-conf", required=True, type=int)
    subparser.add_argument("--low-conf-cutoff", required=True, type=float)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--batch-size", type=int, default=10)

    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", required=True)
    subparser.add_argument("--dev-path", required=True)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--make-trees", action="store_true")
    subparser.add_argument("--erase-labels", action="store_true")

    subparser = subparsers.add_parser("random-constituents")
    subparser.set_defaults(callback=collect_random_constituents)
    subparser.add_argument("--parses", type=str, required=True)


    subparser = subparsers.add_parser("fine-tune-confidence")
    subparser.set_defaults(callback=fine_tune_confidence)
    subparser.add_argument("--model-path-base", required=True)

    subparser = subparsers.add_parser("parse")
    subparser.set_defaults(callback=parse_sentences)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--file-path", required=True)

    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", required=True)
    subparser.add_argument("--dev-path", required=True)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--make-trees", action="store_true")
    subparser.add_argument("--erase-labels", action="store_true")

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
