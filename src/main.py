import argparse
import itertools
import os.path
import pickle
import random
import time
from collections import namedtuple

import dynet as dy
import numpy as np
from scipy import stats
import evaluate
import parse
import trees
import vocabulary
from trees import InternalParseNode, LeafParseNode, ParseNode
from collections import defaultdict

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
        parse, _ = parser.span_parser(tokens)
        parse = parse.convert().linearize()
        parses.append(parse)
    with open('parses.txt', 'w') as f:
        f.write('\n'.join(parses))


def run_span_picking(args):
    parser, _ = load_or_create_model(args, parses_for_vocab=None)
    parses = load_parses(args.trees_path)
    pick_spans_for_annotations(args.annotation_type,
                               parser,
                               parses,
                               args.expt_name,
                               seen=set(),
                               append_to_file_path=None,
                               fraction=1,
                               num_low_conf= 10 ** 8,
                               low_conf_cutoff=args.low_conf_cutoff,
                               high_conf_cutoff=args.high_conf_cutoff)


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





def pick_spans_for_annotations(annotation_type,
                               parser,
                               parses,
                               expt_name,
                               append_to_file_path,
                               fraction,
                               num_low_conf,
                               seen,
                               low_conf_cutoff=0.005,
                               pseudo_label_cutoff=2):
    if not os.path.exists(expt_name):
        os.mkdir(expt_name)

    if annotation_type == "incorrect-spans":
        collect_incorrect_spans = True
        print("Collecting all incorrect spans")
    else:
        assert annotation_type == "uncertainty"
        collect_incorrect_spans = False

    low_confidence_labels = []
    high_confidence_labels = []
    for sentence_number, gold in enumerate(parses):
        if random.random() > fraction:
            continue
        dy.renew_cg()
        if sentence_number % 10000 == 0:
            print(sentence_number, len(low_confidence_labels), len(high_confidence_labels))

        sentence = [(leaf.tag, leaf.word) for leaf in gold.leaves]
        _low_confidence, _high_confidence = parser.return_spans_and_uncertainties(sentence,
                                                                                  sentence_number,
                                                                                  gold,
                                                                                  collect_incorrect_spans,
                                                                                  low_conf_cutoff,
                                                                                  pseudo_label_cutoff,
                                                                                  seen)
        chosen_labels = []
        _low_confidence.sort(key=lambda label: - label['entropy'])
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
        high_confidence_labels.extend(_high_confidence)
        if sentence_number == 0:
            package(low_confidence_labels, os.path.join(expt_name, "test.txt"))

    stats_file_path = os.path.join(expt_name, 'stats.txt')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    package(low_confidence_labels, os.path.join(expt_name, timestr + "-low_confidence_labels.txt"))
    package(high_confidence_labels, os.path.join(expt_name, timestr + "-high_confidence_labels.txt"))
    random.shuffle(low_confidence_labels)
    low_confidence_labels = low_confidence_labels[:int(num_low_conf)]
    # chosen_low_confidence_labels, sentence_number_to_on_spans = pick_spans(low_confidence_labels, num_low_conf, {})
    # chosen_high_confidence_labels, _ = pick_spans(high_confidence_labels, num_high_confidence, sentence_number_to_on_spans)

    # incorrect_high_confidence_count = 0
    # for label in chosen_high_confidence_labels:
    #     sentence_number = label['sentence_number']
    #     gold = parses[sentence_number]
    #     oracle_label = gold.oracle_label(label['left'], label['right'])
    #     incorrect_high_confidence_count += oracle_label != label['label']
    #
    # with open(stats_file_path, 'a') as f:
    #     line = '{} of {} ({}%) high confidence labels are incorrect at {}. Using {} high confidence and {} low confidence labels.\n'.format(
    #         incorrect_high_confidence_count,
    #         len(chosen_high_confidence_labels),
    #         incorrect_high_confidence_count / max(len(chosen_high_confidence_labels), 1),
    #         timestr,
    #     len(chosen_high_confidence_labels),
    #     len(chosen_low_confidence_labels))
    #     f.write(line)
    if append_to_file_path is not None:
        low_confidence_labels.sort(key=lambda label: label['sentence_number'])
        package(low_confidence_labels, append_to_file_path, append=True)
        # package(chosen_high_confidence_labels, append_to_file_path, append=True)


def load_training_spans(args, parser):
    sentence_number_to_data = {}
    with open(os.path.join(args.expt_name, "span_labels.txt"), "r") as f:
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

    temp = trees.load_trees(os.path.join(args.expt_name, "active_learning.trees"))
    temp = list(zip(range(len(temp)), temp))
    annotations_treebank = []
    for sentence_number, tree in temp:
        if sentence_number in sentence_number_to_data:
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            annotations_treebank.append((sentence_number, sentence))

    empty_label_index = parser.label_vocab.index(())
    for sentence_number, sentence in annotations_treebank:
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
    return annotations_treebank, sentence_number_to_data


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

        if args.print_vocabs:
            print_vocabulary("Tag", tag_vocab)
            print_vocabulary("Word", word_vocab)
            print_vocabulary("Label", label_vocab)

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
                results.append((entropy, label_probabilities[gold_label_index, cur_index], predicted_label_index == gold_label_index))
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
        entropies, probabilities, accuracies = zip(*temp)
        output_str += str(np.mean(entropies)) + ' ' + str(np.mean(probabilities)) + ' ' + str(np.mean(accuracies)) + '\n'
        results = results[increment:]
    file_path = os.path.join(expt_name, 'dev_entropy.txt')
    with open(file_path, 'a+') as f:
        f.write(output_str)









def run_training_on_spans(args):
    return_code = os.system('echo "test"')
    assert return_code == 0
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    train_parse = load_parses(args.train_path)

    parser, model = load_or_create_model(args, train_parse)

    train_sentence_number_to_annotations = {}
    train_sentence_number_and_sentence = []
    for sentence_number, parse in enumerate(train_parse):
        sentence_number = "train_" + str(sentence_number)
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        train_sentence_number_and_sentence.append((sentence_number, sentence))
        span_to_gold_label = get_all_spans(parse)
        data = []
        for (left, right), oracle_label in span_to_gold_label.items():
            oracle_label_index = parser.label_vocab.index(oracle_label)
            data.append(label_nt(left=left, right=right, oracle_label_index=oracle_label_index))
        train_sentence_number_to_annotations[sentence_number] = data
    print("Loaded {:,} training examples.".format(len(train_parse)))

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
            dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            predicted, label_probabilities = parser.span_parser(sentence)

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
            return False, dev_fscore

    total_batch_loss = prev_total_batch_loss = None
    active_learning_parses = load_parses(os.path.join(args.expt_name, "active_learning.trees"))
    if args.annotation_type == "none":
        annotated_sentence_number_and_sentence, annotated_sentence_number_to_annotations = load_training_spans(args, parser)
        all_sentence_number_and_sentence = train_sentence_number_and_sentence + annotated_sentence_number_and_sentence
        train_sentence_number_to_annotations.update(annotated_sentence_number_to_annotations)
        all_sentence_number_to_annotations = train_sentence_number_to_annotations
        # TODO update seen
    else:
        seen = set()
    return_code = os.system('echo "test"')
    assert return_code == 0
    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        is_best, dev_score = check_dev()
        perf_summary = '\n' + '-' * 40 + '\n' + str(dev_score) + '\n'
        with open("performance.txt", "a+") as f:
            f.write(perf_summary)
        return_code = os.system("date >> performance.txt")
        assert return_code == 0
        return_code = os.system(
            "wc -l {}/span_labels.txt >> performance.txt".format(args.expt_name))
        assert return_code == 0

        if epoch == 1:
            is_best = False

        print("Total batch loss", total_batch_loss, "Prev batch loss", prev_total_batch_loss)
        if args.annotation_type != "none" and not is_best:
            print("Adding more training data")
            pick_spans_for_annotations(args.annotation_type, parser, active_learning_parses,
                                       args.expt_name,
                                       os.path.join(args.expt_name, "span_labels.txt"),
                                       seen=seen,
                                       fraction=1.0,
                                       num_low_conf=args.num_low_conf,
                                       low_conf_cutoff=float(args.low_conf_cutoff),
                                       pseudo_label_cutoff=2)
            annotated_sentence_number_and_sentence, annotated_sentence_number_to_annotations = \
                load_training_spans(args, parser)
            all_sentence_number_and_sentence = train_sentence_number_and_sentence + annotated_sentence_number_and_sentence
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
            np.random.shuffle(all_sentence_number_and_sentence)
            epoch_start_time = time.time()
            annotation_index = 0
            batch_number = 0
            total_batch_loss = 0
            num_trees = len(all_sentence_number_and_sentence)
            print("Number of trees involved", num_trees)
            while annotation_index < num_trees - args.batch_size:
                dy.renew_cg()
                batch_losses = []

                for _ in range(args.batch_size):
                    sentence_number, sentence = all_sentence_number_and_sentence[annotation_index]
                    annotation_index += 1
                    if args.make_trees:
                        loss = parser.train_on_partial_annotation_make_trees(
                            sentence,
                            all_sentence_number_to_annotations[sentence_number]
                        )
                    else:
                        loss = parser.train_on_partial_annotation(
                            sentence,
                            all_sentence_number_to_annotations[sentence_number]
                        )
                    batch_losses.append(loss)
                    total_processed += 1
                    current_processed += 1


                batch_loss = dy.average(batch_losses)
                batch_loss_value = batch_loss.scalar_value()
                total_batch_loss += batch_loss_value
                batch_loss.backward()
                trainer.update()
                batch_number += 1

                print(
                    "epoch {:,} "
                    "batch {:,}/{:,} "
                    "processed {:,} "
                    "batch-loss {:.4f} "
                    "epoch-elapsed {} "
                    "total-elapsed {}".format(
                        epoch,
                        batch_number,
                        int(np.ceil(num_trees / args.batch_size)),
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
            predicted, _ = parser.span_parser(sentence)
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
                _, loss = parser.span_parser(sentence, tree)
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


def run_test(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)
    print("Loading test trees from {}...".format(args.test_path))
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
    for tree in test_parse:
        if len(test_predicted) % 100 == 0:
            print(len(test_predicted))
            dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        predicted, _ = parser.span_parser(sentence, is_train=False, gold=tree, optimal=args.optimal)
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
    subparser.add_argument("--test-path", default="data/23.auto.clean")
    subparser.add_argument("--separate-left-right", action="store_true")
    subparser.add_argument("--erase-labels", action="store_true")
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--optimal", action="store_true")

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
    subparser.add_argument("--num-low-conf", default=None, type=int)
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
