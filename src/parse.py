import functools

import dynet as dy
import numpy as np
import math
from trees import LeafParseNode, InternalParseNode, ParseNode
from pulp import *
from main import label_nt
from trees import InternalParseNode, LeafParseNode
from scipy import stats
import math
import random
import collections
from main import get_important_spans, get_all_spans
from main import check_overlap

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"


def resolve_conflicts_optimaly(chosen_spans):
    for index_a, (start_a, end_a, on_score_a, off_score_a, _) in enumerate(chosen_spans):
        for index_b, (start_b, end_b, on_score_b, off_score_b, _) in list(
                enumerate(chosen_spans))[index_a + 1:]:
            if start_a < start_b < end_a < end_b or start_b < start_a < end_b < end_a:
                option_a = chosen_spans[:index_a] + chosen_spans[index_a + 1:]
                result_a, score_a = resolve_conflicts_optimaly(option_a)
                score_a += off_score_a + on_score_b
                option_b = chosen_spans[:index_b] + chosen_spans[index_b + 1:]
                result_b, score_b = resolve_conflicts_optimaly(option_b)
                score_b += on_score_a + off_score_b
                if score_a > score_b:
                    return result_a, score_a
                else:
                    return result_b, score_b
    return chosen_spans, 0


def resolve_conflicts_greedily(chosen_spans):
    conflicts_exist = True
    while conflicts_exist:
        conflicts_exist = False
        for index_a, (start_a, end_a, on_score_a, off_score_a, _) in enumerate(chosen_spans):
            for index_b, (start_b, end_b, on_score_b, off_score_b, _) in list(
                    enumerate(chosen_spans))[
                                                                         index_a + 1:]:
                if start_a < start_b < end_a < end_b or start_b < start_a < end_b < end_a:
                    conflicts_exist = True
                    if off_score_a + on_score_b < off_score_b + on_score_a:
                        del chosen_spans[index_b]
                    else:
                        del chosen_spans[index_a]
                    break
    return chosen_spans, None


# Does not assume that chosen_spans do not overlap. Throws an error if
def construct_tree_from_spans(span_to_label, sentence):
    used = set()

    def helper(left, right):
        used.add((left, right))
        if (left, right) in span_to_label:
            label = span_to_label[(left, right)]
            assert label != ()
        else:
            assert left != 0 or right != len(sentence)
            label = ()

        if right - left == 1:
            tag, word = sentence[left]
            tree = LeafParseNode(left, tag, word)
            if label:
                tree = InternalParseNode(label, [tree])
            return [tree]

        argmax_split = left + 1
        for split in range(right - 1, left, -1):
            if (left, split) in span_to_label:
                argmax_split = split
                break
        assert left < argmax_split < right, (left, argmax_split, right)
        left_trees = helper(left, argmax_split)
        right_trees = helper(argmax_split, right)
        children = left_trees + right_trees
        if label:
            children = [InternalParseNode(label, children)]
        return children

    trees = helper(0, len(sentence))
    for span in span_to_label.keys():
        assert span in used, (span, 'not in used spans')
    assert len(trees) == 1
    return trees[0]


def optimal_parser(label_log_probabilities_np,
                   span_to_index,
                   sentence,
                   empty_label_index,
                   label_vocab,
                   gold=None):
    def choose_consistent_spans():
        greedily_chosen_spans = []
        rank = 1
        gold_parse_log_likelihood = 0
        confusion_matrix = collections.defaultdict(int)
        for (start, end), span_index in span_to_index.items():
            off_score = label_log_probabilities_np[empty_label_index, span_index]
            on_score = np.max(label_log_probabilities_np[1:,
                              span_index])  # math.log(max(1 - math.exp(off_score), 10 ** -8))
            if on_score > off_score or (start == 0 and end == len(sentence)):
                label_index = label_log_probabilities_np[1:, span_index].argmax() + 1
                greedily_chosen_spans.append((start, end, on_score, off_score, label_index))
            else:
                label_index = empty_label_index
            if gold is not None:
                oracle_label = gold.oracle_label(start, end)
                oracle_label_index = label_vocab.index(oracle_label)
                label = label_vocab.value(label_index)
                oracle_label_log_probability = label_log_probabilities_np[
                    oracle_label_index, span_index]
                gold_parse_log_likelihood += oracle_label_log_probability
                confusion_matrix[(label, oracle_label)] += 1

        choices, _ = resolve_conflicts_optimaly(greedily_chosen_spans)
        span_to_label = {}
        predicted_parse_log_likelihood = np.sum(label_log_probabilities_np[empty_label_index, :])
        adjusted = label_log_probabilities_np - label_log_probabilities_np[empty_label_index, :]
        for choice in choices:
            span_to_label[(choice[0], choice[1])] = label_vocab.value(choice[4])
            span_index = span_to_index[(choice[0], choice[1])]
            predicted_parse_log_likelihood += adjusted[choice[4], span_index]
        num_spans_forced_off = len(greedily_chosen_spans) - len(span_to_label)
        return span_to_label, (
            predicted_parse_log_likelihood, confusion_matrix, num_spans_forced_off, rank)

    span_to_label, additional_info = choose_consistent_spans()
    tree = construct_tree_from_spans(span_to_label, sentence)
    return tree, additional_info


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
        self.elmo_weights = self.model.parameters_from_numpy(
            np.array([0.16397782, 0.67511874, 0.02329052]), name='elmo-averaging-weights')
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
            tag_embedding_dim + word_embedding_dim + 1024,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)

        self.dropout = dropout
        self.empty_label = ()
        self.empty_label_index = self.label_vocab.index(self.empty_label)

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def _featurize_sentence(self, sentence, is_train, elmo_embeddings, cur_word_index):
        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()
        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and (np.random.rand() < 1 / (1 + count))):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            if tag == START or tag == STOP:
                concatenated_embeddings = [tag_embedding, word_embedding, dy.zeros(1024)]
            else:
                elmo_weights = dy.parameter(self.elmo_weights)
                embedding = dy.sum_dim(dy.cmult(elmo_weights, elmo_embeddings[cur_word_index]), [0])
                concatenated_embeddings = [tag_embedding, word_embedding, embedding]
                cur_word_index += 1
            embeddings.append(dy.concatenate(concatenated_embeddings))
        return self.lstm.transduce(embeddings)

    def _get_span_encoding(self, left, right, lstm_outputs):
        forward = (
            lstm_outputs[right][:self.lstm_dim] -
            lstm_outputs[left][:self.lstm_dim])
        backward = (
            lstm_outputs[left + 1][self.lstm_dim:] -
            lstm_outputs[right + 1][self.lstm_dim:])
        return dy.concatenate([forward, backward])

    def _encodings_to_label_log_probabilities(self, encodings, lmbd=None, alpha=None):
        label_scores = self.f_label(dy.concatenate_to_batch(encodings))
        label_scores_reshaped = dy.reshape(label_scores, (self.label_vocab.size, len(encodings)))

        # if alpha is not None:
        #     temp = dy.abs(dy.reshape(alpha[0], (1, 1)))
        #     label_scores_reshaped = dy.cmult(dy.logistic(dy.cmult(label_scores_reshaped, temp) + alpha[1]), lmbd) + alpha[2]
        # 990.51641846]] [ 0.03124614  4.00097179 -9.43100834
        # label_scores_reshaped = dy.logistic(label_scores_reshaped * 0.03124614 + 4.00097179) * 990.51641846 - 9.43100834
        return dy.log_softmax(label_scores_reshaped)

    def train_on_partial_annotation(self, sentence, annotations, elmo_vecs, cur_word_index):
        if len(annotations) == 0:
            return dy.zeros(1)
        lstm_outputs = self._featurize_sentence(sentence, is_train=True, elmo_embeddings=elmo_vecs,
                                                cur_word_index=cur_word_index)

        encodings = []
        for annotation in annotations:
            assert 0 <= annotation.left < annotation.right <= len(sentence), \
                (0, annotation.left, annotation.right, len(sentence))
            encoding = self._get_span_encoding(annotation.left, annotation.right, lstm_outputs)
            encodings.append(encoding)

        label_log_probabilities = self._encodings_to_label_log_probabilities(encodings)
        total_loss = dy.zeros(1)
        for index, annotation in reversed(list(enumerate(annotations))):
            loss = - label_log_probabilities[annotation.oracle_label_index][index]
            total_loss = total_loss + loss
        return total_loss


    def get_distribution_for_kbest(self, sentence, elmo_embeddings, cur_word_index):
        assert self.empty_label_index == 0, self.empty_label_index
        lstm_outputs = self._featurize_sentence(sentence, is_train=False,
                                                elmo_embeddings=elmo_embeddings,
                                                cur_word_index=cur_word_index)
        encodings = []
        span_to_index = {}
        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span_to_index[(start, end)] = len(encodings)
                encodings.append(self._get_span_encoding(start, end, lstm_outputs))

        label_log_probabilities_np = self._encodings_to_label_log_probabilities(
            encodings).npvalue()
        non_constituent_probabilities = np.exp(
            label_log_probabilities_np[self.empty_label_index, :])
        label_log_probabilities_np = np.log(np.array([non_constituent_probabilities, 1 - non_constituent_probabilities]))
        return (label_log_probabilities_np, span_to_index)

    def produce_parse_forest(self, sentence, required_probability_mass):
        lstm_outputs = self._featurize_sentence(sentence, is_train=False)
        encodings = []
        spans = []
        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                spans.append((start, end))
                encodings.append(self._get_span_encoding(start, end, lstm_outputs))
        label_scores = self.f_label(dy.concatenate_to_batch(encodings))
        label_scores_reshaped = dy.reshape(label_scores,
                                           (self.label_vocab.size, len(encodings)))
        label_probabilities_np = dy.softmax(label_scores_reshaped).npvalue()
        span_to_labels = {}
        forest_prob_mass = 1
        for index, span in enumerate(spans):
            distribution = list(enumerate(label_probabilities_np[:, index]))
            distribution.sort(key=lambda x: - x[1])
            total_probability = 0
            labels = []
            while total_probability < required_probability_mass:
                (label_index, probability) = distribution.pop()
                labels.append(self.label_vocab.values[label_index])
                total_probability += probability
            forest_prob_mass *= total_probability
            span_to_labels[span] = labels
        return span_to_labels, forest_prob_mass

    def return_spans_and_uncertainties(self,
                                       sentence,
                                       sentence_number,
                                       gold,
                                       use_oracle,
                                       low_conf_cutoff,
                                       pseudo_label_cutoff,
                                       seen):
        spans = [span for span in get_all_spans(gold).keys() if
                 (span, sentence_number) not in seen]
        if len(spans) == 0:
            return []
        lstm_outputs = self._featurize_sentence(sentence, is_train=False)
        encodings = []
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
                non_constituent_probability=distribution[0]
            )
            if use_oracle:
                oracle_label_index = self.label_vocab.index(oracle_label)
                if oracle_label_index != predicted_label_index and distribution[
                    oracle_label_index] > 0.01:
                    annotation_request['label'] = oracle_label
                    low_confidence_labels.append(annotation_request)
            elif max(distribution) > pseudo_label_cutoff and (
                            distribution[
                                self.empty_label_index] < 0.001 or random.random() < 0.001):
                annotation_request['label'] = predicted_label
                high_confidence_labels.append(annotation_request)
            elif low_conf_cutoff < entropy:
                annotation_request['label'] = oracle_label
                low_confidence_labels.append(annotation_request)
        return low_confidence_labels, high_confidence_labels

    def aggressive_annotation(self,
                              sentence,
                              sentence_number,
                              span_to_gold_label,
                              low_conf_cutoff,
                              seen):
        if len(span_to_gold_label) == 0:
            return []  # , []
        lstm_outputs = self._featurize_sentence(sentence, is_train=False)
        encodings = []
        spans = span_to_gold_label.keys()
        for (start, end) in spans:
            encodings.append(self._get_span_encoding(start, end, lstm_outputs))
        label_scores = self.f_label(dy.concatenate_to_batch(encodings))
        label_scores_reshaped = dy.reshape(label_scores,
                                           (self.label_vocab.size, len(encodings)))
        label_probabilities_np = dy.softmax(label_scores_reshaped).npvalue()
        low_confidence_labels = []
        # high_confidence_labels = []
        on_labels = []
        for index, (start, end) in list(enumerate(spans)):
            distribution = label_probabilities_np[:, index]
            entropy = stats.entropy(distribution)
            oracle_label = span_to_gold_label[(start, end)]
            annotation_request = dict(
                sentence_number=sentence_number,
                left=start,
                right=end,
                entropy=entropy,
                non_constituent_probability=distribution[0],
                label=oracle_label
            )
            if (start, end) in seen:
                del span_to_gold_label[(start, end)]
                continue
            if low_conf_cutoff < entropy and distribution[self.empty_label_index] < 0.5:
                # annotation_request['label'] = oracle_label
                low_confidence_labels.append(annotation_request)
            elif entropy < 10 ** -5 and distribution[self.empty_label_index] > 0.99:
                del span_to_gold_label[(start, end)]
                # if entropy > 10 ** -7:
                #     high_confidence_labels.append(annotation_request)
            if np.max(distribution) > distribution[self.empty_label_index]:
                on_labels.append(annotation_request)

        for index, label_a in enumerate(on_labels):
            span_a = (label_a['left'], label_a['right'])
            for label_b in on_labels[index + 1:]:
                span_b = (label_b['left'], label_b['right'])
                if check_overlap(span_a, span_b):
                    label_a['entropy'] = 10
                    low_confidence_labels.append(label_a)
                    label_b['entropy'] = 10
                    low_confidence_labels.append(label_b)

        return low_confidence_labels  # , high_confidence_labels

    def compute_label_distributions(self, sentence, is_train, elmo_embeddings, cur_word_index):
        lstm_outputs = self._featurize_sentence(sentence, is_train=is_train,
                                                elmo_embeddings=elmo_embeddings,
                                                cur_word_index=cur_word_index)
        encodings = []
        span_to_index = {}
        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span_to_index[(start, end)] = len(encodings)
                encodings.append(self._get_span_encoding(start, end, lstm_outputs))
        label_log_probabilities = self._encodings_to_label_log_probabilities(encodings)
        return span_to_index, label_log_probabilities

    def span_parser(self, sentence, is_train, elmo_embeddings, cur_word_index, gold=None):
        if gold is not None:
            assert isinstance(gold, ParseNode)

        span_to_index, label_log_probabilities = self.compute_label_distributions(sentence,
                                                                                  is_train,
                                                                                  elmo_embeddings,
                                                                                  cur_word_index)

        total_loss = dy.zeros(1)
        if is_train:
            for start in range(0, len(sentence)):
                for end in range(start + 1, len(sentence) + 1):
                    gold_label = gold.oracle_label(start, end)
                    gold_label_index = self.label_vocab.index(gold_label)
                    if gold_label_index != 0:
                        gold_label_index = 1
                    index = span_to_index[(start, end)]
                    total_loss -= label_log_probabilities[gold_label_index][index]
            return None, total_loss
        else:
            label_log_probabilities_np = label_log_probabilities.npvalue()
            tree, additional_info = optimal_parser(label_log_probabilities_np,
                                                   span_to_index,
                                                   sentence,
                                                   self.empty_label_index,
                                                   self.label_vocab,
                                                   gold)
            return tree, additional_info, dy.exp(label_log_probabilities).npvalue()

    def fine_tune_confidence(self, sentence, lmbd, alpha, elmo_embeddings, cur_word_index, gold):
        lstm_outputs = self._featurize_sentence(sentence, is_train=False,
                                                elmo_embeddings=elmo_embeddings,
                                                cur_word_index=cur_word_index)
        encodings = []
        span_to_index = {}
        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span_to_index[(start, end)] = len(encodings)
                encodings.append(self._get_span_encoding(start, end, lstm_outputs))
        label_log_probabilities = self._encodings_to_label_log_probabilities(encodings, lmbd=lmbd,
                                                                             alpha=alpha)

        total_loss = dy.zeros(1)
        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                gold_label = gold.oracle_label(start, end)
                gold_label_index = self.label_vocab.index(gold_label)
                index = span_to_index[(start, end)]
                total_loss -= label_log_probabilities[gold_label_index][index]
        return total_loss
