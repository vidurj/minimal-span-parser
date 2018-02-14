import argparse
import hashlib
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
import spacy



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
        print('populating params from 40k_elmo.model')
        parser.f_label.param_collection().populate('40k_elmo.model', '/ffn')
        parser.word_embeddings.populate('40k_elmo.model', '/word-embeddings')
        parser.tag_embeddings.populate('40k_elmo.model', '/tag-embeddings')
        # parser.mlp.populate('2-predict-pos.model', '/mlp')
        parser.lstm.param_collection().populate('40k_elmo.model', '/lstm')
        # parser.elmo_weights.populate('predict-pos.model', '/elmo-weights')
        # parser.f_encoding.weights[0].populate('predict-pos.model', '/encoding-weights')
        # parser.f_encoding.biases[0].populate('predict-pos.model', '/encoding-biases')
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
    return None
    # results = []
    # for sentence_number, gold in enumerate(dev_parses):
    #     dy.renew_cg()
    #     sentence = [(leaf.tag, leaf.word) for leaf in gold.leaves]
    #     label_probabilities = matrices[sentence_number]
    #     cur_index = -1
    #     for start in range(0, len(sentence)):
    #         for end in range(start + 1, len(sentence) + 1):
    #             cur_index += 1
    #             gold_label = gold.oracle_label(start, end)
    #             gold_label_index = parser.label_vocab.index(gold_label)
    #             entropy = span_to_entropy[(sentence_number, start, end)]
    #             predicted_label_index = np.argmax(label_probabilities[:, cur_index])
    #             results.append((entropy, math.log(label_probabilities[gold_label_index, cur_index] + 10 ** -8), predicted_label_index == gold_label_index))
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # results.sort(key=lambda x: x[0])
    # file_path = os.path.join(expt_name, timestr + '-entropy-results.pickle')
    # with open(file_path, 'wb') as f:
    #     pickle.dump(results, f)
    #
    # num_buckets = 20
    # increment = int(len(results) / num_buckets)
    # output_str = "\n"
    # for i in range(num_buckets):
    #     temp = results[:increment]
    #     entropies, log_probabilities, accuracies = zip(*temp)
    #     output_str += str(np.mean(entropies)) + ' ' + str(np.mean(log_probabilities)) + ' ' + str(np.mean(accuracies)) + '\n'
    #     results = results[increment:]
    # file_path = os.path.join(expt_name, 'dev_entropy.txt')
    # with open(file_path, 'a+') as f:
    #     f.write(output_str)





def run_train(args):
    return_code = os.system('cp -r src {}/'.format(args.expt_name))
    assert return_code == 0
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    if not args.predict_pos:
        assert args.train_path.endswith('.clean') and args.dev_path.endswith('.clean')

    all_parses = load_parses(args.train_path)

    with open(os.path.join(args.expt_name, "train_tree_indices.txt"), 'r') as f:
        train_tree_indices = [int(x) for x in f.read().strip().splitlines()]
    assert max(train_tree_indices) < len(all_parses)
    print('training on', len(train_tree_indices), 'full trees')

    parser, model = load_or_create_model(args, all_parses)
    assert parser.use_elmo != args.no_elmo
    assert parser.predict_pos == args.predict_pos

    # parser.mlp.save('2-predict-pos.model', '/mlp', append=False)
    # parser.word_embeddings.save('2-predict-pos.model', '/word-embedding', append=True)
    # print('-' * 100)


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

    dev_sentence_embeddings = h5py.File('../wsj-dev.hdf5', 'r')

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        # total_loglikelihood = 0
        matrices = []
        fill_span_to_entropy = len(span_to_entropy) == 0
        num_correct = 0
        total = 0
        for sentence_number, tree in enumerate(dev_treebank):
            if sentence_number % 100 == 0:
                dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            elmo_embeddings_np = dev_sentence_embeddings[str(sentence_number)][:, :, :]
            assert elmo_embeddings_np.shape[1] == len(sentence)
            elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
            predicted, (_, c, t) = parser.span_parser(sentence, is_train=False, elmo_embeddings=elmo_embeddings)
            num_correct += c
            total += t

            dev_predicted.append(predicted.convert())

        print('pos accuracy', num_correct / float(total + 1))

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

    total_batch_loss = prev_total_batch_loss = None

    return_code = os.system('echo "test"')
    assert return_code == 0
    train_sentence_embeddings = h5py.File('../wsj-train.hdf5', 'r')
    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        if epoch > 1:
            save_latest_model(args.model_path_base, parser)
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
        prev_total_batch_loss = total_batch_loss
        epoch_start_time = time.time()
        num_batches = 0
        total_batch_loss = 0
        np.random.shuffle(train_tree_indices)
        for start_tree_index in range(0, len(train_tree_indices), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            for offset in range(args.batch_size):
                if start_tree_index + offset < len(train_tree_indices):
                    sentence_number = train_tree_indices[start_tree_index + offset]
                    gold = all_parses[sentence_number]
                    sentence = [(leaf.tag, leaf.word) for leaf in gold.leaves]
                    elmo_embeddings_np = train_sentence_embeddings[str(sentence_number)][:, :, :]
                    assert elmo_embeddings_np.shape[1] == len(sentence), (len(sentence), elmo_embeddings_np.shape)
                    elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
                    loss = parser.span_parser(sentence, is_train=True, elmo_embeddings=elmo_embeddings, gold=gold)
                    batch_losses.append(loss)
                    total_processed += 1
                    current_processed += 1

            if len(batch_losses) > 0:
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
                        len(all_parses) / args.batch_size,
                        total_processed,
                        batch_loss_value,
                        format_elapsed(epoch_start_time),
                        format_elapsed(start_time),
                    )
                )


def evaluate_bro_co(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    dirs = ['cf', 'cg', 'ck', 'cl', 'cm', 'cn', 'cp', 'cr']
    for dir in dirs:
        print('-' * 100)
        print(dir)
        input_file = '../brown/' + dir + '/' + dir + '.all.mrg'
        expt_name = args.expt_name + '/' + dir
        if not os.path.exists(expt_name):
            os.mkdir(expt_name)
        cleaned_corpus_path = trees.cleanup_text(input_file)
        treebank = trees.load_trees(cleaned_corpus_path, strip_top=True, filter_none=True)
        sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves] for tree in treebank]
        tokenized_lines = [' '.join([word for pos, word in sentence]) for sentence in sentences]
        embedding_file = compute_embeddings(tokenized_lines, expt_name)

        model = dy.ParameterCollection()
        [parser] = dy.load(args.model_path_base, model)

        dev_predicted = []
        for tree_index, tree in enumerate(treebank):
            if tree_index % 100 == 0:
                print(tree_index)
                dy.renew_cg()
            sentence = sentences[tree_index]
            embeddings_np = embedding_file[str(tree_index)][:, :, :]
            assert embeddings_np.shape[1] == len(sentence), (embeddings_np.shape[1], len(sentence))
            embeddings = dy.inputTensor(embeddings_np)
            predicted, _ = parser.span_parser(sentence, is_train=False,
                                              elmo_embeddings=embeddings)
            dev_predicted.append(predicted.convert())

        dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, dev_predicted,
                                                   args=args,
                                                   erase_labels=True,
                                                   name="without-labels",
                                                   expt_name=expt_name)
        print("dev-fscore without labels", dev_fscore_without_labels)

        dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, dev_predicted,
                                                   args=args,
                                                   erase_labels=True,
                                                   flatten=True,
                                                   name="without-label-flattened",
                                                   expt_name=expt_name)
        print("dev-fscore without labels and flattened", dev_fscore_without_labels)

        dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, dev_predicted,
                                                   args=args,
                                                   erase_labels=False,
                                                   flatten=True,
                                                   name="flattened",
                                                   expt_name=expt_name)
        print("dev-fscore with labels and flattened", dev_fscore_without_labels)

        test_fscore = evaluate.evalb('EVALB/', treebank, dev_predicted, args=args,
                                     name="regular",
                                                   expt_name=expt_name)

        print("regular", test_fscore)


def run_test_qbank(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    treebank = trees.load_trees('questionbank/all_qb_trees.txt')

    if args.stanford_split == 'true':
        print('using stanford split')
        split_to_indices = {
            'train': list(range(0, 1000)) + list(range(2000, 3000)),
            'dev': list(range(1000, 1500)) + list(range(3000, 3500)),
            'test': list(range(1500, 2000)) + list(range(3500, 4000))
        }
    else:
        print('not using stanford split')
        split_to_indices = {
            'train': range(0, 2000),
            'dev': range(2000, 3000),
            'test': range(3000, 4000)
        }

    indices = split_to_indices[args.split]
    test_treebank = [treebank[index] for index in indices]
    test_embeddings = []
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
        predicted, _ = parser.span_parser(sentence, is_train=False,
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



def parse_file(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print(parser.label_vocab.counts)


    with open(args.input_file, 'r') as f:
        lines = f.read().splitlines()

    nlp = spacy.load('en')
    sentences = [[(None, token.text) for token in nlp(line)] for line in lines]
    tokenized_lines = [' '.join([word for pos, word in sentence]) for sentence in sentences]
    elmo_embeddings_file_path = compute_embeddings(tokenized_lines, args.expt_name)

    cur_word_index = 0
    output_string = ""
    for index, sentence in enumerate(sentences):
        if index % 100 == 0:
            dy.renew_cg()
        elmo_embeddings_np = elmo_embeddings_file_path[str(index)][:, :, :]
        assert elmo_embeddings_np.shape[1] == len(sentence)
        elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
        if args.simple:
            parse, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=elmo_embeddings)
            output_string += parse.convert().linearize() + '\n'
        else:
            (label_log_probabilities_np, span_to_index) = parser.get_distribution_for_kbest(sentence,
                                                                                            elmo_embeddings,
                                                                                            cur_word_index)
            parses, scores = kbest((sentence, int(args.num_parses), label_log_probabilities_np, span_to_index))
            sentence = [word for pos, word in sentence]
            output_string += '\n' + ' '.join(sentence) + '\n\n'
            for parse, score in zip(parses, scores):
                output_string += str(math.exp(score)) + '\n' + parse.linearize() + '\n'
            output_string += '*' * 70 + '\n'
        cur_word_index += len(sentence)
    with open(args.expt_name + '/output.txt', 'w') as f:
        f.write(output_string)

def compute_embeddings(tokenized_lines, expt_name):
    if not os.path.exists(expt_name):
        os.mkdir(expt_name)

    tokenized_sentences_file_path = expt_name + '/tokenized_sentences.txt'

    normalized_lines = []
    for line in tokenized_lines:
        cleaned_line = ''
        for word in line.split():
            if word == '-LRB-' or word == 'LRB':
                word = '('
            elif word == '-RRB-' or word == 'RRB':
                word = ')'
            elif word == '-LCB-' or word == 'LCB':
                word = '{'
            elif word == '-RCB-' or word == 'RCB':
                word = '}'
            elif word == '`':
                word = "'"
            elif word == "''":
                word = '"'

            if '\\' in word:
                print('-' * 100)
                print(word)
                print('-' * 100)
            cleaned_line += word + ' '
        normalized_lines.append(cleaned_line.strip())
    with open(tokenized_sentences_file_path, 'w') as f:
        f.write('\n'.join(normalized_lines))
    elmo_embeddings_file_path = expt_name + '/elmo.hdf5'
    print(elmo_embeddings_file_path)
    if not os.path.exists(elmo_embeddings_file_path):
        generate_elmo_vectors = '/home/vidurj/miniconda3/envs/allennlp/bin/python3 ' \
                                '-m allennlp.run elmo {} {} --all'.format(
            tokenized_sentences_file_path,
            elmo_embeddings_file_path
        )
        print(generate_elmo_vectors)
        return_code = os.system(generate_elmo_vectors)
        assert return_code == 0, return_code
    else:
        print('Using precomputed embeddings at', elmo_embeddings_file_path)
    embedding_file = h5py.File(elmo_embeddings_file_path, 'r')
    return embedding_file


def test_on_brackets(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    with open(args.input_file, 'r') as f:
        lines = f.read().splitlines()

    nlp = spacy.load('en')

    processed_lines = []
    sentence_number_to_bracketings = defaultdict(set)
    for sentence_number, line in enumerate(lines):
        tokens = [token.text for token in nlp(line)]
        open_bracket_indices = []
        sentence = []
        print(line)
        print('***')
        constituent_spans = set()
        for token in tokens:
            if token == '[' or token == '{':
                open_bracket_indices.append(len(sentence))
            elif token == ']' or token == '}':
                is_constituent = token == ']'
                span = (open_bracket_indices.pop(), len(sentence))
                temp = sentence + ["*" for _ in range(5)]
                print(temp[span[0]: span[1]], is_constituent)
                sentence_number_to_bracketings[sentence_number].add((span, is_constituent))
                if is_constituent:
                    constituent_spans.add(span)
            else:
                sentence.append(token)

        for start in range(len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span = (start, end)
                for constituent_span in constituent_spans:
                    if check_overlap(span, constituent_span):
                        assert span not in constituent_spans
                        sentence_number_to_bracketings[sentence_number].add((span, False))
        print('-' * 40)
        assert len(open_bracket_indices) == 0, open_bracket_indices
        processed_lines.append(' '.join(sentence))
    # print(sentence_number_to_bracketings)

    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)
    embedding_file = compute_embeddings(processed_lines, args.expt_name + '/test')
    sentences = [[(None, word) for word in line.split()] for line in processed_lines]
    cur_word_index = 0
    num_correct = 0
    num_wrong = 0
    num_perfect_sentences = 0
    num_sentences = 0
    output_string = ''
    for sentence_number, sentence in enumerate(sentences):
        elmo_embeddings_np = embedding_file[str(sentence_number)][:, :, :]
        assert elmo_embeddings_np.shape[1] == len(sentence)
        elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
        lstm_outputs = parser._featurize_sentence(sentence, is_train=False,
                                                elmo_embeddings=elmo_embeddings)
        cur_word_index += len(sentence)
        encodings = []
        span_to_index = {}
        for ((start, end), is_constituent) in sentence_number_to_bracketings[sentence_number]:
            span_to_index[(start, end)] = len(encodings)
            encodings.append(parser._get_span_encoding(start, end, lstm_outputs))

        if len(encodings) == 0:
            print('skipping sentence', sentence_number)
            continue

        label_scores = parser.f_label(dy.rectify(parser.f_encoding(dy.concatenate_to_batch(encodings))))
        label_scores_reshaped = dy.reshape(label_scores,
                                           (parser.label_vocab.size, len(encodings)))
        label_probabilities = dy.softmax(label_scores_reshaped)


        is_perfect = True
        for (span, is_constituent) in sentence_number_to_bracketings[sentence_number]:
            span_index = span_to_index[span]
            non_constituent_prob_np = label_probabilities[parser.empty_label_index][span_index].scalar_value()
            if is_constituent and non_constituent_prob_np < 0.5:
                num_correct += 1
            elif not is_constituent and non_constituent_prob_np > 0.5:
                num_correct += 1
            else:
                is_perfect = False
                num_wrong += 1
        if not is_perfect:
            print(sentence_number)
        num_perfect_sentences += is_perfect
        num_sentences += 1

        parse, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=elmo_embeddings)
        output_string += parse.convert().linearize() + '\n'


    print('accuracy', num_correct / float(num_correct + num_wrong), num_correct, num_wrong)
    print('num sentences', num_sentences, 'num perfect sentences', num_perfect_sentences, 'fraction', num_perfect_sentences / float(num_sentences))
    results = str(args) + '\n' + str(('accuracy', num_correct / float(num_correct + num_wrong), num_correct, num_wrong)) + \
              str(('num sentences', num_sentences, 'num perfect sentences', num_perfect_sentences, 'fraction', num_perfect_sentences / float(num_sentences)))
    with open(args.expt_name + '/trained_test_results.txt', 'w') as f:
        f.write(results)
    with open(args.expt_name + '/parses.txt', 'w') as f:
        f.write(output_string)

def train_on_brackets(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    with open(args.input_file, 'r') as f:
        lines = f.read().splitlines()

    nlp = spacy.load('en')

    processed_lines = []
    sentence_number_to_bracketings = defaultdict(set)
    for sentence_number, line in enumerate(lines):
        tokens = [token.text for token in nlp(line)]
        open_bracket_indices = []
        sentence = []
        print(line)
        print('***')
        constituent_spans = set()
        for token in tokens:
            if token == '[' or token == '{':
                open_bracket_indices.append(len(sentence))
            elif token == ']' or token == '}':
                is_constituent = token == ']'
                span = (open_bracket_indices.pop(), len(sentence))
                temp = sentence + ["*" for _ in range(5)]
                print(temp[span[0]: span[1]], is_constituent)
                sentence_number_to_bracketings[sentence_number].add((span, is_constituent))
                if is_constituent:
                    constituent_spans.add(span)
            else:
                sentence.append(token)

        for start in range(len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span = (start, end)
                for constituent_span in constituent_spans:
                    if check_overlap(span, constituent_span):
                        assert span not in constituent_spans
                        sentence_number_to_bracketings[sentence_number].add((span, False))
        print('-' * 40)
        assert len(open_bracket_indices) == 0, open_bracket_indices
        processed_lines.append(' '.join(sentence))
    print(sentence_number_to_bracketings)

    wsj_train = load_parses('data/train.trees')
    parser, model = load_or_create_model(args, wsj_train)
    trainer = dy.AdamTrainer(model)
    embedding_file = compute_embeddings(processed_lines, args.expt_name)

    wsj_embedding_file = h5py.File('../wsj-train.hdf5', 'r')
    sentences = [[(None, word) for word in line.split()] for line in processed_lines]

    wsj_indices = []

    for epoch in itertools.count(start=1):
        dy.renew_cg()


        loss = dy.zeros(1)
        num_correct = 0
        num_wrong = 0
        for sentence_number, sentence in enumerate(sentences):
            elmo_embeddings_np = embedding_file[str(sentence_number)][:, :, :]
            assert elmo_embeddings_np.shape[1] == len(sentence)
            elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
            lstm_outputs = parser._featurize_sentence(sentence, is_train=True,
                                                    elmo_embeddings=elmo_embeddings)
            encodings = []
            span_to_index = {}
            for ((start, end), is_constituent) in sentence_number_to_bracketings[sentence_number]:
                span_to_index[(start, end)] = len(encodings)
                encodings.append(parser._get_span_encoding(start, end, lstm_outputs))

            if len(encodings) == 0:
                continue

            label_scores = parser.f_label(dy.rectify(parser.f_encoding(dy.concatenate_to_batch(encodings))))
            label_scores_reshaped = dy.reshape(label_scores,
                                               (parser.label_vocab.size, len(encodings)))
            label_probabilities = dy.softmax(label_scores_reshaped)



            for (span, is_constituent) in sentence_number_to_bracketings[sentence_number]:
                span_index = span_to_index[span]
                non_constituent_prob_np = label_probabilities[parser.empty_label_index][span_index].scalar_value()
                if is_constituent:
                    print(' '.join([word for pos, word in sentence[span[0]: span[1]]]))
                    loss -= dy.log(1 - label_probabilities[parser.empty_label_index][span_index] + 10 ** -7)
                else:
                    loss -= dy.log(label_probabilities[parser.empty_label_index][span_index] + 10 ** -7)
                if is_constituent and non_constituent_prob_np < 0.5:
                    num_correct += 1
                elif not is_constituent and non_constituent_prob_np > 0.5:
                    num_correct += 1
                else:
                    num_wrong += 1


        print(loss.scalar_value())
        print('accuracy', num_correct / float(num_correct + num_wrong), num_correct, num_wrong)

        batch_losses = [loss]

        for _ in range(50):
            if not wsj_indices:
                wsj_indices = list(range(39832))
                random.shuffle(wsj_indices)

            index = wsj_indices.pop()
            tree = wsj_train[index]
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            wsj_embeddings_np = wsj_embedding_file[str(index)][:, :, :]
            assert wsj_embeddings_np.shape[1] == len(sentence)
            wsj_embeddings = dy.inputTensor(wsj_embeddings_np)
            loss = parser.span_parser(sentence, is_train=True, gold=tree,
                                         elmo_embeddings=wsj_embeddings)
            batch_losses.append(loss)


        batch_loss = dy.average(batch_losses)
        batch_loss_value = batch_loss.scalar_value()
        batch_loss.backward()
        trainer.update()

        print(batch_loss_value)
        print('-' * 100)

        if epoch % 10 == 0:
            save_latest_model(args.model_path_base, parser)


def save_latest_model(model_path_base, parser):
    latest_model_path = "{}_latest_model".format(model_path_base)
    for ext in [".data", ".meta"]:
        path = latest_model_path + ext
        if os.path.exists(path):
            print("Removing previous model file {}...".format(path))
            os.remove(path)

    print("Saving new model to {}...".format(latest_model_path))
    dy.save(latest_model_path, [parser])


def run_train_question_bank_stanford_split(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    all_parses = load_parses('questionbank/all_qb_trees.txt')


    print("Loaded {:,} trees.".format(len(all_parses)))

    tentative_stanford_train_indices = list(range(0, 1000)) + list(range(2000, 3000))
    stanford_dev_indices = list(range(1000, 1500)) + list(range(3000, 3500))
    stanford_test_indices = list(range(1500, 2000)) + list(range(3500, 4000))


    dev_and_test_sentences = set()
    dev_parses = []
    dev_sentence_number_to_sentence_and_word_index = {}
    word_index = 0
    for index in stanford_dev_indices:
        parse = all_parses[index]
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        dev_sentence_number_to_sentence_and_word_index[len(dev_parses)] = (sentence, word_index)
        word_index += len(sentence)
        dev_parses.append(parse)
        dev_and_test_sentences.add(tuple(sentence))
    final_dev_word_index = word_index

    dev_treebank = [parse.convert() for parse in dev_parses]

    for index in stanford_test_indices:
        parse = all_parses[index]
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        dev_and_test_sentences.add(tuple(sentence))


    train_parses = []
    train_sentence_number_to_sentence_and_word_index = {}
    word_index = 0
    stanford_train_indices = []
    for index in tentative_stanford_train_indices:
        parse = all_parses[index]
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        if tuple(sentence) not in dev_and_test_sentences:
            train_sentence_number_to_sentence_and_word_index[len(train_parses)] = (sentence, word_index)
            word_index += len(sentence)
            train_parses.append(parse)
            stanford_train_indices.append(index)
    final_train_word_index = word_index


    train_embeddings = []
    dev_embeddings = []
    with h5py.File('question_bank_elmo_embeddings.hdf5', 'r') as h5f:
        for index in stanford_train_indices:
            train_embeddings.append(h5f[str(index)][:, :, :])

        for index in stanford_dev_indices:
            dev_embeddings.append(h5f[str(index)][:, :, :])

    train_embeddings_np = np.swapaxes(np.concatenate(train_embeddings, axis=1), axis1=0,
                                      axis2=1)
    assert final_train_word_index == len(train_embeddings_np), (final_train_word_index, len(train_embeddings_np))

    dev_embeddings_np = np.swapaxes(np.concatenate(dev_embeddings, axis=1), axis1=0,
                                    axis2=1)
    assert final_dev_word_index == len(dev_embeddings_np), (final_dev_word_index, len(dev_embeddings_np))



    print("We have {:,} train trees.".format(len(train_parses)))

    wsj_train = load_parses('data/train.trees')

    parser, model = load_or_create_model(args, train_parses + wsj_train)
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()


    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for index, tree in enumerate(dev_treebank):
            if len(dev_predicted) % 100 == 0:
                dy.renew_cg()
                dev_embeddings = dy.inputTensor(dev_embeddings_np)

            sentence, cur_word_index = dev_sentence_number_to_sentence_and_word_index[index]
            predicted, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=dev_embeddings, cur_word_index=cur_word_index)
            dev_predicted.append(predicted.convert())

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

    wsj_indices = []
    if args.resample == 'true':
        assert args.num_samples == 'false'
        print('training on resampled data')
        tree_indices = [random.randint(0, len(train_parses) - 1) for _ in range(len(train_parses))]
    elif args.num_samples != 'false':
        print('restricting to', args.num_samples, 'samples')
        tree_indices = list(range(len(train_parses)))
        random.shuffle(tree_indices)
        tree_indices = tree_indices[:int(args.num_samples)]
    else:
        print('training on original data')
        tree_indices = list(range(len(train_parses)))

    if args.train_on_wsj == 'true':
        print('training on wsj')
    else:
        print('not training on wsj')

    with open(args.expt_name + '/train_tree_indices.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in tree_indices]))

    for epoch in itertools.count(start=1):
        np.random.shuffle(tree_indices)
        epoch_start_time = time.time()

        for start_index in range(0, len(tree_indices), args.batch_size):
            dy.renew_cg()
            train_embeddings = dy.inputTensor(train_embeddings_np)
            batch_losses = []
            for index in tree_indices[start_index: start_index + args.batch_size]:
                tree = train_parses[index]
                sentence, cur_word_index = train_sentence_number_to_sentence_and_word_index[index]
                _, loss = parser.span_parser(sentence, is_train=True, gold=tree, elmo_embeddings=train_embeddings, cur_word_index=cur_word_index)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            if args.train_on_wsj == 'true':
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
                    start_index // args.batch_size,
                    int(np.ceil(len(tree_indices) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

        if epoch % 100 == 0:
            check_dev()

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
            predicted, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=dev_embeddings, cur_word_index=cur_word_index)
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
            predicted, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=dev_embeddings, cur_word_index=cur_word_index)
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
            # string_to_score = {}
            for (label_index, action_score) in actions:
                for (children, children_score) in children_options:
                    option_score = action_score + children_score
                    if label_index != 0:
                        label = labels[label_index]
                        tree = InternalParseNode(label, children)
                        option = [tree]
                    else:
                        option = children
                    # option_string = ''.join([compute_string(tree) for tree in option])
                    # if option_string in string_to_score and option_score <= string_to_score[option_string]:
                    #     continue

                    # string_to_score[option_string] = option_score
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
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

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
    text = args.expt_name + args.test_path + args.model_path_base
    hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    print(hash, str(args))
    data_file_path = 'hash-' + hash + '.pickle'
    if os.path.isfile(data_file_path):
        print('loading from cache', data_file_path)
        with open(data_file_path, 'rb') as f:
            data = pickle.load(f)
    else:
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
        with open(data_file_path, 'wb') as f:
            pickle.dump(data, f)

    responses = pool.map(kbest, data, chunksize=int(math.floor(len(data) / num_cpus)))
    tree_loglikelihoods = []
    predicted_acc = []
    gold_acc = []
    num_trees = []
    for (tree, (predicted_trees, loglikelihoods)) in zip(test_parses, responses):
        # assert len(predicted_trees_and_scores) == args.num_trees, (sentence, len(predicted_trees_and_scores))
        # predicted, additional_info, _ = parser.span_parser(sentence, is_train=False, gold=tree)
        gold_treebank_tree = tree.convert()
        predicted_acc.extend(predicted_trees)
        gold_acc.extend([gold_treebank_tree for _ in predicted_trees])
        tree_loglikelihoods.extend(loglikelihoods)
        assert len(predicted_trees) == len(loglikelihoods)
        num_trees.append(len(predicted_trees))

    evaluate.evalb(args.evalb_dir, gold_acc, predicted_acc, args=args,
                                 name="bestk", flatten=True, erase_labels=True)
    file_path = os.path.join(args.expt_name, 'bestk-output.txt')
    with open(file_path, 'r') as f:
        text = f.read().strip().splitlines()
    flag = False
    forest_sizes = list(range(1, 500))
    forest_sizes = [x for x in forest_sizes if x <= args.num_trees]
    forest_sizes.sort()
    forest_size_to_scores = {}
    for k in forest_sizes:
        forest_size_to_scores[k] = []

    forest_size_to_ks = {}
    for k in forest_sizes:
        forest_size_to_ks[k] = []

    thresholds = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    thresholds.sort()
    threshold_to_scores = defaultdict(list)
    threshold_to_ks = defaultdict(list)

    all_loglikelihoods = [x for _, loglikelihoods in responses for x in loglikelihoods[1:]]
    all_loglikelihoods.sort(key = lambda x: -x)
    k_to_optimal_cutoff = {}
    for k in forest_sizes:
        index = (k - 1) * len(test_parses) + 1
        if index < 0:
            print(k, index, 'index less than zero')
            continue
        elif index >= len(all_loglikelihoods):
            index = len(all_loglikelihoods) - 1
            print(k, index, 'index larger than list')

        k_to_optimal_cutoff[k] = math.exp(all_loglikelihoods[index])

    k_to_scores = defaultdict(list)
    k_to_counts = defaultdict(list)
    k_to_max_tree_number = defaultdict(lambda: -1)
    buffer = []
    buffer_probability = 0
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
            probability = math.exp(tree_loglikelihoods.pop(0))
            old_buffer_probability = buffer_probability
            buffer_probability += probability

            best_f1 = max(buffer, key=lambda x: x[3])

            if len(buffer) == 1:
                for k in forest_sizes:
                    k_to_scores[k].append(best_f1)
                    k_to_counts[k].append(len(buffer))
            else:
                for k in forest_sizes:
                    if k in k_to_optimal_cutoff and probability > k_to_optimal_cutoff[k]:
                        k_to_scores[k][-1] = best_f1
                        k_to_counts[k][-1] = len(buffer)

            if len(buffer) in forest_size_to_scores:
                forest_size_to_scores[len(buffer)].append(best_f1)
                forest_size_to_ks[len(buffer)].append(len(buffer))

            for threshold in thresholds:
                if old_buffer_probability < threshold <= buffer_probability:
                    threshold_to_scores[threshold].append(best_f1)
                    threshold_to_ks[threshold].append(len(buffer))


            if len(buffer) == num_trees[cur_tree_number]:
                # for threshold in thresholds:
                #     if buffer_probability < threshold:
                #         threshold_to_scores[threshold].append(best_f1)
                #         threshold_to_ks[threshold].append(len(buffer))
                for forest_size, scores in forest_size_to_scores.items():
                    if len(buffer) < forest_size:
                        scores.append(best_f1)
                        forest_size_to_ks[forest_size].append(len(buffer))
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
              'num sentences:', len(best_scores))
        with open(name + '.pickle', 'wb') as f:
            pickle.dump(best_scores, f)

    for size in forest_sizes:
        if size not in k_to_optimal_cutoff:
            continue
        print(size)
        # assert len(forest_size_to_scores[size]) == len(test_parses)
        # assert len(forest_size_to_ks[size]) == len(test_parses)
        print('avg k', np.mean(k_to_counts[size]))
        print_stats(k_to_scores[size], 'k_equals_' + str(size))
        print('-' * 50)

    print('*' * 100)

    for size in forest_sizes:
        print(size)
        # assert len(forest_size_to_scores[size]) == len(test_parses)
        # assert len(forest_size_to_ks[size]) == len(test_parses)
        print('avg k', np.mean(forest_size_to_ks[size]))
        print_stats(forest_size_to_scores[size], 'k_equals_' + str(size))
        print('-' * 50)

    print('*' * 100)

    for probability in thresholds:
        print('threshold', probability)
        # assert len(threshold_to_ks[probability]) == len(test_parses)
        # assert len(threshold_to_scores[probability]) == len(test_parses)
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
    if file_name == 'test':
        sentence_embeddings = h5py.File('../wsj-test.hdf5', 'r')
    elif file_name == 'dev':
        sentence_embeddings = h5py.File('../wsj-dev.hdf5', 'r')
    else:
        assert file_name == 'train', file_name
        sentence_embeddings = h5py.File('../wsj-train.hdf5', 'r')

    test_treebank = trees.load_trees(args.test_path)

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
    num_correct = 0
    total = 0
    for tree_index, tree in enumerate(test_parse):
        if tree_index % 100 == 0:
            print(tree_index)
            dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        elmo_embeddings_np = sentence_embeddings[str(tree_index)][:, :, :]
        assert elmo_embeddings_np.shape[1] == len(sentence), (elmo_embeddings_np.shape[1], len(sentence))
        elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
        predicted, (additional_info, c, t) = parser.span_parser(sentence, is_train=False, elmo_embeddings=elmo_embeddings, gold=tree)
        num_correct += c
        total += t
        rank = additional_info[3]
        ranks.append(rank)
        total_log_likelihood += additional_info[-1]
        test_predicted.append(predicted.convert())
        # total_turned_off += _turned_off
        # for k, v in _confusion_matrix.items():
        #     if k in total_confusion_matrix:
        #         total_confusion_matrix[k] += v
        #     else:
        #         total_confusion_matrix[k] = v
    print('pos accuracy', num_correct / total)
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


def compute_kbest_f1_bagged_qb(args):

    sizes = [1, 5, 10, 15, 20, 50, 100, 200, 300, 400, 500]
    sizes.sort()
    sizes = [x for x in sizes if x <= args.num_trees]



    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    print("Loading model from {}...".format(args.model_path_base))
    meta_model = dy.ParameterCollection()
    parsers = []
    num_parsers = 10
    for i in range(1, num_parsers + 1):
        model = meta_model.add_subcollection('model_' + str(i))
        relevant_files = [f[:-5] for f in os.listdir(str(i)) if f.startswith('model') and 'latest' not in f and f.endswith('.data')]
        model_file = max(relevant_files, key=lambda x: len(x))
        print('loading model for', i, 'from', file)
        [parser] = dy.load(model_file, model)
        parsers.append(parser)

    assert args.split == 'dev', args.split
    test_embeddings = []

    if args.split == 'train':
        indices = list(range(0, 1000)) + list(range(2000, 3000))
    elif args.split == 'dev':
        indices = list(range(1000, 1500)) + list(range(3000, 3500))
    else:
        assert args.split == 'test', args.split
        indices = list(range(1500, 2000)) + list(range(3500, 4000))

    with h5py.File('question_bank_elmo_embeddings.hdf5', 'r') as h5f:
        for index in indices:
            test_embeddings.append(h5f[str(index)][:, :, :])

    all_parses = load_parses('questionbank/all_qb_trees.txt')
    test_parses = [all_parses[index] for index in indices]

    test_embeddings_np = np.swapaxes(np.concatenate(test_embeddings, axis=1), axis1=0, axis2=1)

    print("Parsing test sentences...")
    data_from_parsers = [[] for _ in range(11)]
    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool()
    cur_word_index = 0
    for index, tree in enumerate(test_parses):
        if index % 100 == 0:
            dy.renew_cg()
            test_embeddings = dy.inputTensor(test_embeddings_np)
            h5f.close()
            print(index)
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        meta_label_probabilities = None
        for parser_number in range(num_parsers):
            parser = parsers[parser_number]
            span_to_index, label_log_probabilities = parser.compute_label_distributions(
                sentence,
                is_train=False,
                elmo_embeddings=test_embeddings,
                cur_word_index=cur_word_index)
            label_log_probabilities_np = label_log_probabilities.npvalue()
            data_from_parsers[parser_number].append((sentence, args.num_trees, label_log_probabilities_np, span_to_index))
            if parser_number == 0:
                meta_label_probabilities = np.exp(label_log_probabilities_np)
            else:
                meta_label_probabilities += np.exp(label_log_probabilities_np)

        meta_label_probabilities /= len(parsers)
        meta_label_log_probabilities = np.log(meta_label_probabilities)
        data_from_parsers[num_parsers].append((sentence, args.num_trees, meta_label_log_probabilities, span_to_index))
        cur_word_index += len(sentence)


    responses = []
    for data in data_from_parsers:
        responses.append(pool.map(kbest, data, chunksize=int(math.floor(len(data) / num_cpus))))
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


def collect_mistakes_on_qb_bagged(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    meta_model = dy.ParameterCollection()
    parsers = []
    for i in range(1, 11):
        model = meta_model.add_subcollection('model' + str(i))
        relevant_files = [f[:-5] for f in os.listdir(str(i)) if f.startswith('model') and 'latest' not in f and f.endswith('.data')]
        model_file = max(relevant_files, key=lambda x: len(x))
        print('loading model for', i, 'from', model_file)
        [parser] = dy.load(str(i) + '/' + model_file, model)
        parsers.append(parser)

    assert args.split == 'dev', args.split
    test_embeddings = []

    if args.split == 'train':
        indices = list(range(0, 1000)) + list(range(2000, 3000))
    elif args.split == 'dev':
        indices = list(range(1000, 1500)) + list(range(3000, 3500))
    else:
        assert args.split == 'test', args.split
        indices = list(range(1500, 2000)) + list(range(3500, 4000))

    with h5py.File('question_bank_elmo_embeddings.hdf5', 'r') as h5f:
        for index in indices:
            test_embeddings.append(h5f[str(index)][:, :, :])

    all_parses = load_parses('questionbank/all_qb_trees.txt')
    test_parses = [all_parses[index] for index in indices]

    test_embeddings_np = np.swapaxes(np.concatenate(test_embeddings, axis=1), axis1=0, axis2=1)

    errors = []
    cur_word_index = 0
    total_log_likelihood = 0
    predicted_treebank = []
    for dev_index, tree in enumerate(test_parses):
        if dev_index % 100 == 0:
            dy.renew_cg()
            test_embeddings = dy.inputTensor(test_embeddings_np)
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        sentence_str = ' '.join([word for tag, word in sentence])
        words = [word for pos, word in sentence]
        span_to_index, _label_log_probabilities = parsers[0].compute_label_distributions(sentence,
                                                                                  is_train=False,
                                                                                  elmo_embeddings=test_embeddings,
                                                                                  cur_word_index=cur_word_index)
        meta_label_probabilities = np.exp(_label_log_probabilities.npvalue())
        for parser in parsers[1:]:
            _, label_probabilities = parser.compute_label_distributions(
                sentence,
                is_train=False,
                elmo_embeddings=test_embeddings,
                cur_word_index=cur_word_index)
            meta_label_probabilities += np.exp(_label_log_probabilities.npvalue())

        meta_label_probabilities /= len(parsers)
        meta_label_log_probabilities = np.log(meta_label_probabilities)
        cur_word_index += len(sentence)
        span_to_label = get_all_spans(tree)

        predicted_tree, additional_info = parse.optimal_parser(meta_label_log_probabilities,
                                                      span_to_index,
                                                      sentence,
                                                      parser.empty_label_index,
                                                      parser.label_vocab,
                                                      tree)
        predicted_treebank.append(predicted_tree.convert())
        total_log_likelihood += additional_info[-1]
        predicted_parse_string = predicted_tree.convert().linearize()
        gold_parse_string = tree.convert().linearize()

        worst_error_prob = None
        worst_error_string = None
        worst_error_gold_label = None
        for span, label in span_to_label.items():
            label_index = parser.label_vocab.index(label)
            span_index = span_to_index[span]
            nc_prob = meta_label_probabilities[parser.empty_label_index, span_index]

            if label_index == parser.empty_label_index:
                prob = nc_prob
            else:
                prob = 1 - nc_prob
            if worst_error_prob is None or prob < worst_error_prob:
                worst_error_prob = prob
                worst_error_string = ' '.join(words[span[0]:span[1]])
                if label == ():
                    worst_error_gold_label = 'NC'
                else:
                    worst_error_gold_label = ' '.join(label)


        errors.append((sentence_str, gold_parse_string, predicted_parse_string, worst_error_string, worst_error_gold_label, str(worst_error_prob)))

    print('total loglikelihood', total_log_likelihood)
    keys = ['sentence', 'gold parse', 'pred parse', 'span text', 'span gold label', 'span gold label probability']
    errors.sort(key=lambda x: x[0])
    error_string = ""
    for lines in errors:
        formatted_lines = [x + ': ' + y for x, y in zip(keys, lines)]
        error_string += '\n'.join(formatted_lines) + '\n***\n'
    with open(args.expt_name + '-errors.txt', 'w') as f:
        f.write(error_string)
    dev_fscore = evaluate.evalb('EVALB/', [tree.convert() for tree in test_parses], predicted_treebank, args=args,
                                name="dev-regular")
    print(dev_fscore)


def collect_mistakes_on_qb(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    assert args.split == 'dev', args.split
    test_embeddings = []

    if args.split == 'train':
        indices = list(range(0, 1000)) + list(range(2000, 3000))
    elif args.split == 'dev':
        indices = list(range(1000, 1500)) + list(range(3000, 3500))
    else:
        assert args.split == 'test', args.split
        indices = list(range(1500, 2000)) + list(range(3500, 4000))

    with h5py.File('question_bank_elmo_embeddings.hdf5', 'r') as h5f:
        for index in indices:
            test_embeddings.append(h5f[str(index)][:, :, :])

    all_parses = load_parses('questionbank/all_qb_trees.txt')
    test_parses = [all_parses[index] for index in indices]

    test_embeddings_np = np.swapaxes(np.concatenate(test_embeddings, axis=1), axis1=0, axis2=1)

    errors = []
    cur_word_index = 0
    total_loglikelihood = 0
    predicted_treebank = []
    for dev_index, tree in enumerate(test_parses):
        if dev_index % 100 == 0:
            dy.renew_cg()
            test_embeddings = dy.inputTensor(test_embeddings_np)
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        sentence_str = ' '.join([word for tag, word in sentence])
        words = [word for pos, word in sentence]
        span_to_index, label_log_probabilities = parser.compute_label_distributions(sentence,
                                                                                  is_train=False,
                                                                                  elmo_embeddings=test_embeddings,
                                                                                  cur_word_index=cur_word_index)
        label_log_probabilities = label_log_probabilities.npvalue()
        cur_word_index += len(sentence)
        span_to_label = get_all_spans(tree)

        predicted_tree, additional_info = parse.optimal_parser(label_log_probabilities,
                                                      span_to_index,
                                                      sentence,
                                                      parser.empty_label_index,
                                                      parser.label_vocab,
                                                      tree)
        predicted_treebank.append(predicted_tree.convert())
        total_loglikelihood += additional_info[-1]
        predicted_parse_string = predicted_tree.convert().linearize()
        gold_parse_string = tree.convert().linearize()

        worst_error_prob = None
        worst_error_string = None
        worst_error_gold_label = None
        for span, label in span_to_label.items():
            label_index = parser.label_vocab.index(label)
            span_index = span_to_index[span]
            nc_prob = math.exp(label_log_probabilities[parser.empty_label_index, span_index])

            if label_index == parser.empty_label_index:
                prob = nc_prob
            else:
                prob = 1 - nc_prob
            if worst_error_prob is None or prob < worst_error_prob:
                worst_error_prob = prob
                worst_error_string = ' '.join(words[span[0]:span[1]])
                if label == ():
                    worst_error_gold_label = 'NC'
                else:
                    worst_error_gold_label = ' '.join(label)


        errors.append((sentence_str, gold_parse_string, predicted_parse_string, worst_error_string, worst_error_gold_label, str(worst_error_prob)))
    print('total loglikelihood', total_loglikelihood)
    keys = ['sentence', 'gold parse', 'pred parse', 'span text', 'span gold label', 'span gold label probability']
    errors.sort(key=lambda x: x[0])
    error_string = ""
    for lines in errors:
        formatted_lines = [x + ': ' + y for x, y in zip(keys, lines)]
        error_string += '\n'.join(formatted_lines) + '\n***\n'
    with open(args.expt_name + '-errors.txt', 'w') as f:
        f.write(error_string)
    dev_fscore = evaluate.evalb('EVALB/', [tree.convert() for tree in test_parses],
                                predicted_treebank, args=args,
                                name="dev-regular")
    print(dev_fscore)


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
    ks = [0 for _ in range(parser.label_vocab.size + 10)]
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
        sentence_str = ' '.join([word for tag, word in sentence])
        words = [word for pos, word in sentence]
        span_to_index, label_log_probabilities = parser.compute_label_distributions(sentence,
                                                                                  is_train=False,
                                                                                  elmo_embeddings=elmo_embeddings,
                                                                                  cur_word_index=cur_word_index)
        label_log_probabilities = label_log_probabilities.npvalue()
        cur_word_index += len(sentence)
        span_to_label = get_all_spans(tree)

        predicted_tree, _ = parse.optimal_parser(label_log_probabilities,
                                                      span_to_index,
                                                      sentence,
                                                      parser.empty_label_index,
                                                      parser.label_vocab,
                                                      tree)

        predicted_parse_string = predicted_tree.convert().linearize()
        gold_parse_string = tree.convert().linearize()

        worst_error_prob = None
        worst_error_string = None
        worst_error_gold_label = None
        for span, label in span_to_label.items():
            label_index = parser.label_vocab.index(label)
            span_index = span_to_index[span]
            nc_prob = math.exp(label_log_probabilities[parser.empty_label_index, span_index])
            if math.exp(np.max(label_log_probabilities[:, span_index])) < 0.5:
                k = np.sum(label_log_probabilities[:, span_index] > label_log_probabilities[label_index, span_index])
                ks[int(k)] += 1
            if label_index == parser.empty_label_index:
                prob = nc_prob
            else:
                prob = 1 - nc_prob
            if worst_error_prob is None or prob < worst_error_prob:
                worst_error_prob = prob
                worst_error_string = ' '.join(words[span[0]:span[1]])
                if label == ():
                    worst_error_gold_label = 'NC'
                else:
                    worst_error_gold_label = ' '.join(label)


        errors.append((sentence_str, gold_parse_string, predicted_parse_string, worst_error_string, worst_error_gold_label, str(worst_error_prob)))
    print(ks)
    keys = ['sentence', 'gold parse', 'pred parse', 'span text', 'span gold label', 'span gold label probability']
    errors.sort(key=lambda x: x[0])
    error_string = ""
    for lines in errors:
        formatted_lines = [x + ': ' + y for x, y in zip(keys, lines)]
        error_string += '\n'.join(formatted_lines) + '\n***\n'
    with open(args.expt_name + '-errors.txt', 'w') as f:
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


    subparser = subparsers.add_parser("parse-bro-co")
    subparser.set_defaults(callback=evaluate_bro_co)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--input-file", required=True)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("train-on-brackets")
    subparser.set_defaults(callback=train_on_brackets)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--input-file", required=True)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("test-on-brackets")
    subparser.set_defaults(callback=test_on_brackets)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--input-file", required=True)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("parse-file")
    subparser.set_defaults(callback=parse_file)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--num-parses", type=int, default=10)
    subparser.add_argument("--input-file", required=True)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--simple", action="store_true")

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

    subparser = subparsers.add_parser("train-question-bank-stanford-split")
    subparser.set_defaults(callback=run_train_question_bank_stanford_split)
    for arg in dynet_args:
        subparser.add_argument(arg)

    subparser.add_argument("--train-on-wsj", required=True)
    subparser.add_argument("--resample", required=True)
    subparser.add_argument("--num-samples", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--batch-size", type=int, default=100)



    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--no-elmo", action="store_true")
    subparser.add_argument("--predict-pos", action="store_true")
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
    subparser.add_argument("--stanford-split", required=True)
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

    subparser = subparsers.add_parser("collect-errors-on-qb")
    subparser.set_defaults(callback=collect_mistakes_on_qb)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--split", default='dev')

    subparser = subparsers.add_parser("collect-errors-on-qb-bagged")
    subparser.set_defaults(callback=collect_mistakes_on_qb_bagged)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--split", default='dev')

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

    subparser = subparsers.add_parser("random-constituents")
    subparser.set_defaults(callback=collect_random_constituents)
    subparser.add_argument("--parses", type=str, required=True)


    subparser = subparsers.add_parser("fine-tune-confidence")
    subparser.set_defaults(callback=fine_tune_confidence)
    subparser.add_argument("--model-path-base", required=True)

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
