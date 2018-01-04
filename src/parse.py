import functools

import dynet as dy
import numpy as np
import math
import trees
from pulp import *
from main import label_nt
from trees import InternalParseNode, LeafParseNode
from scipy import stats
import math
import random
import collections
from main import get_important_spans, get_all_spans

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"


def augment(scores, oracle_index):
    assert isinstance(scores, dy.Expression)
    shape = scores.dim()[0]
    assert len(shape) == 1
    increment = np.ones(shape)
    increment[oracle_index] = 0
    return scores + dy.inputVector(increment)


def check_overlap(span_a, span_b):
    return span_a[0] < span_b[0] < span_a[1] < span_b[1] or \
           span_b[0] < span_a[0] < span_b[1] < span_a[1]


def resolve_conflicts(chosen_spans):
    for index_a, (start_a, end_a, on_score_a, off_score_a, _) in enumerate(chosen_spans):
        for index_b, (start_b, end_b, on_score_b, off_score_b, _) in list(
                enumerate(chosen_spans))[index_a + 1:]:
            if start_a < start_b < end_a < end_b or start_b < start_a < end_b < end_a:
                option_a = chosen_spans[:index_a] + chosen_spans[index_a + 1:]
                result_a, score_a = resolve_conflicts(option_a)
                score_a += off_score_a + on_score_b
                option_b = chosen_spans[:index_b] + chosen_spans[index_b + 1:]
                result_b, score_b = resolve_conflicts(option_b)
                score_b += on_score_a + off_score_b
                if score_a > score_b:
                    return result_a, score_a
                else:
                    return result_b, score_b
    return chosen_spans, 0


def optimal_parser(label_log_probabilities,
                   span_to_index,
                   sentence,
                   empty_label_index,
                   label_vocab,
                   gold=None):
    def construct_trees_from_spans(left, right, chosen_spans):
        if (left, right) in chosen_spans:
            label_index = chosen_spans[(left, right)][4]
            assert label_index != empty_label_index
        else:
            assert left != 0 or right != len(sentence)
            label_index = empty_label_index
        label = label_vocab.value(label_index)

        if right - left == 1:
            tag, word = sentence[left]
            tree = trees.LeafParseNode(left, tag, word)
            if label:
                tree = trees.InternalParseNode(label, [tree])
            return [tree]

        argmax_split = left + 1
        for split in range(right - 1, left, -1):
            if (left, split) in chosen_spans:
                argmax_split = split
                break
        assert left < argmax_split < right, (left, argmax_split, right)
        left_trees = construct_trees_from_spans(left, argmax_split, chosen_spans)
        right_trees = construct_trees_from_spans(argmax_split, right, chosen_spans)
        children = left_trees + right_trees
        if label:
            children = [trees.InternalParseNode(label, children)]
        return children

    def choose_consistent_spans():
        label_log_probabilities_np = label_log_probabilities.npvalue()
        greedily_chosen_spans = []
        parse_log_likelihood = 0
        confusion_matrix = collections.defaultdict(int)
        for (start, end), span_index in span_to_index.items():
            off_score = label_log_probabilities_np[empty_label_index, span_index]
            on_score = math.log(max(1 - math.exp(off_score), 10 ** -5))
            if on_score > off_score or (start == 0 and end == len(sentence)):
                label_index = label_log_probabilities_np[1:, span_index].argmax() + 1
                greedily_chosen_spans.append((start, end, on_score, off_score, label_index))
            else:
                label_index = empty_label_index
            if gold is not None:
                oracle_label = gold.oracle_label(start, end)
                oracle_label_index = label_vocab.index(oracle_label)
                label = label_vocab.value(label_index)
                parse_log_likelihood += label_log_probabilities_np[oracle_label_index, span_index]
                confusion_matrix[(label, oracle_label)] += 1

        choices, _ = resolve_conflicts(greedily_chosen_spans)
        chosen_spans = {}
        for choice in choices:
            chosen_spans[(choice[0], choice[1])] = choice
        num_spans_forced_off = len(greedily_chosen_spans) - len(chosen_spans)
        return chosen_spans, (parse_log_likelihood, confusion_matrix, num_spans_forced_off)

    chosen_spans, additional_info = choose_consistent_spans()
    trees = construct_trees_from_spans(0, len(sentence), chosen_spans)
    assert len(trees) == 1, len(trees)
    return trees[0], additional_info


class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(self.model.add_parameters((next_dim, prev_dim)))
            self.biases.append(self.model.add_parameters(next_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weight = dy.parameter(weight)
            bias = dy.parameter(bias)
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x


class TopDownParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.separate_left_right = True

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split_left = Feedforward(
            self.model, 2 * lstm_dim, [split_hidden_dim], 1)
        if self.separate_left_right:
            self.f_split_right = Feedforward(
                self.model, 2 * lstm_dim, [split_hidden_dim], 1)

        self.dropout = dropout
        self.empty_label = ()
        self.empty_label_index = self.label_vocab.index(self.empty_label)

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def _featurize_sentence(self, sentence, is_train):
        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()
        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))
        return self.lstm.transduce(embeddings)

    def _get_span_encoding(self, left, right, lstm_outputs):
        forward = (
            lstm_outputs[right][:self.lstm_dim] -
            lstm_outputs[left][:self.lstm_dim])
        backward = (
            lstm_outputs[left + 1][self.lstm_dim:] -
            lstm_outputs[right + 1][self.lstm_dim:])
        return dy.concatenate([forward, backward])

    def _encodings_to_label_log_probabilities(self, encodings):
        label_scores = self.f_label(dy.concatenate_to_batch(encodings))
        label_scores_reshaped = dy.reshape(label_scores, (self.label_vocab.size, len(encodings)))
        return dy.log_softmax(label_scores_reshaped)

    def train_on_partial_annotation(self, sentence, annotations):
        lstm_outputs = self._featurize_sentence(sentence, is_train=True)

        encodings = []
        for annotation in annotations:
            encoding = self._get_span_encoding(annotation.left, annotation.right, lstm_outputs)
            encodings.append(encoding)

        label_log_probabilities = self._encodings_to_label_log_probabilities(encodings)
        total_loss = dy.zeros(1)
        for index, annotation in enumerate(annotations):
            total_loss = total_loss - label_log_probabilities[annotation.oracle_label_index][index]
        return total_loss

    def return_spans_and_uncertainties(self,
                                       sentence,
                                       sentence_number,
                                       gold,
                                       use_oracle,
                                       low_conf_cutoff=0.005,
                                       high_conf_cutoff=0.0001):
        lstm_outputs = self._featurize_sentence(sentence, is_train=False)
        encodings = []
        spans = get_all_spans(gold).keys()
        for (start, end) in spans:
            encodings.append(self._get_span_encoding(start, end, lstm_outputs))
        label_scores = self.f_label(dy.concatenate_to_batch(encodings))
        label_scores_reshaped = dy.reshape(label_scores,
                                           (self.label_vocab.size, len(encodings)))
        label_probabilities_np = dy.softmax(label_scores_reshaped).npvalue()
        low_confidence_labels = []
        high_confidence_labels = []
        for index, (start, end) in enumerate(spans):
            distribution = label_probabilities_np[:, index]
            entropy = stats.entropy(distribution)
            oracle_label = gold.oracle_label(start, end)
            predicted_label_index = distribution.argmax()
            predicted_label = self.label_vocab.value(predicted_label_index)
            annotation_request = dict(
                sentence_number=sentence_number,
                left=start,
                right=end,
                entropy=entropy,
                non_constituent_probability=distribution[0],
                oracle_label=oracle_label,
                predicted_label=predicted_label
            )
            if use_oracle:
                oracle_label_index = self.label_vocab.index(oracle_label)
                if oracle_label_index != predicted_label_index:
                    low_confidence_labels.append(annotation_request)
            elif low_conf_cutoff < entropy:
                low_confidence_labels.append(annotation_request)
            if entropy < high_conf_cutoff:
                high_confidence_labels.append(annotation_request)
        return low_confidence_labels, high_confidence_labels

    def span_parser(self, sentence, gold=None, is_train=None, optimal=True):
        if gold is not None:
            assert isinstance(gold, trees.ParseNode)
        if is_train is None and gold is not None:
            is_train = True

        lstm_outputs = self._featurize_sentence(sentence, is_train)

        encodings = []
        span_to_index = {}
        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span_to_index[(start, end)] = len(encodings)
                encodings.append(self._get_span_encoding(start, end, lstm_outputs))
        label_log_probabilities = self._encodings_to_label_log_probabilities(encodings)

        total_loss = dy.zeros(1)
        if is_train:
            for start in range(0, len(sentence)):
                for end in range(start + 1, len(sentence) + 1):
                    gold_label = gold.oracle_label(start, end)
                    gold_label_index = self.label_vocab.index(gold_label)
                    index = span_to_index[(start, end)]
                    total_loss -= label_log_probabilities[gold_label_index][index]
            return None, total_loss
        else:
            tree, additional_info = optimal_parser(label_log_probabilities,
                                                   span_to_index,
                                                   sentence,
                                                   self.empty_label_index,
                                                   self.label_vocab,
                                                   gold)
            return tree, additional_info
