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
deletable_tags = {',', ':', '``', "''", '.'}


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


def optimal_tree_construction(span_to_label, sentence, span_to_on_score):
    conflicting = set()
    for span_a in span_to_label:
        for span_b in span_to_label:
            if check_overlap(span_a, span_b):
                conflicting.add(span_a)
    cache = {}

    def helper(left, right):
        if (left, right) in cache:
            return cache[(left, right)]

        if (left, right) in span_to_label:
            label = span_to_label[(left, right)]
            assert label != ()
        else:
            assert left != 0 or right != len(sentence)
            label = ()

        if right - left == 1:
            tag, word = sentence[left]
            tree = LeafParseNode(left, tag, word)
            score = 0
            if label:
                tree = InternalParseNode(label, [tree])
                score += span_to_on_score[(left, right)]
            return [tree], score

        split_options = []
        for split in range(right - 1, left, -1):
            if (left, split) in span_to_label:
                split_options.append(split)
                if (left, split) not in conflicting:
                    break
            if split == left + 1:
               split_options.append(left + 1)
        assert len(split_options) > 0
        best_option_score = None
        best_option = None
        for split in split_options:
            left_trees, left_score = helper(left, split)
            right_trees, right_score = helper(split, right)
            children = left_trees + right_trees
            score = left_score + right_score
            if label:
                children = [InternalParseNode(label, children)]
                score += span_to_on_score[(left, right)]

            if best_option_score is None or score > best_option_score:
                best_option_score = score
                best_option = children
        response = best_option, best_option_score
        cache[(left, right)] = response
        return response

    trees, _ = helper(0, len(sentence))
    assert (0, len(sentence)) in span_to_label
    assert len(trees) == 1, len(trees)
    return trees[0]


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
                greedily_chosen_spans.append((start, end, on_score - off_score, None, label_index))
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

        #choices, _ = resolve_conflicts_greedily(greedily_chosen_spans)
        span_to_label = {}
        predicted_parse_log_likelihood = np.sum(label_log_probabilities_np[empty_label_index, :])
        adjusted = label_log_probabilities_np - label_log_probabilities_np[empty_label_index, :]
        span_to_on_score = {}
        for choice in greedily_chosen_spans:
            span_to_label[(choice[0], choice[1])] = label_vocab.value(choice[4])
            span_to_on_score[(choice[0], choice[1])] = choice[2]
            span_index = span_to_index[(choice[0], choice[1])]
            predicted_parse_log_likelihood += adjusted[choice[4], span_index]
        num_spans_forced_off = len(greedily_chosen_spans) - len(span_to_label)
        return span_to_label, span_to_on_score, (
            predicted_parse_log_likelihood, confusion_matrix, num_spans_forced_off, rank,
            gold_parse_log_likelihood)

    span_to_label, span_to_on_score, additional_info = choose_consistent_spans()
    tree = optimal_tree_construction(span_to_label, sentence, span_to_on_score)
    return tree, additional_info


class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")
        self.layer_params = []
        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            layer_params = self.model.add_subcollection("Layer" + str(len(self.weights)))
            self.weights.append(layer_params.add_parameters((next_dim, prev_dim)))
            self.biases.append(layer_params.add_parameters(next_dim))
            self.layer_params.append(layer_params)

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
            dropout,
            use_elmo=True,
            predict_pos=True
    ):
        use_elmo = True
        predict_pos = True
        assert predict_pos
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.mlp = self.model.add_subcollection("mlp")

        self.tag_vocab = tag_vocab
        print('tag vocab', tag_vocab.size)
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim
        self.hidden_dim = label_hidden_dim
        self.predict_pos = predict_pos

        lstm_input_dim = word_embedding_dim
        if use_elmo:
            self.elmo_weights = self.model.parameters_from_numpy(
                np.array([0.19608361, 0.53294581, -0.00724584]), name='elmo-averaging-weights')
            lstm_input_dim += 1024
        if not predict_pos:
            lstm_input_dim += tag_embedding_dim

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            lstm_input_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        if not predict_pos:
            self.tag_embeddings = self.model.add_lookup_parameters(
                (tag_vocab.size, tag_embedding_dim))
            self.f_label = Feedforward(self.mlp, 2 * lstm_dim, [label_hidden_dim],
                                       label_vocab.size)
        else:
            self.f_encoding = Feedforward(
                self.mlp, 2 * lstm_dim, [], label_hidden_dim)

            self.f_label = Feedforward(
                self.mlp, label_hidden_dim, [], label_vocab.size)

            self.f_tag = Feedforward(
                self.mlp, label_hidden_dim, [], tag_vocab.size)

        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.use_elmo = use_elmo

        self.dropout = dropout
        self.empty_label = ()
        self.empty_label_index = self.label_vocab.index(self.empty_label)

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def _featurize_sentence(self, sentence, is_train, elmo_embeddings):
        # assert len(sentence) == elmo_embeddings.dim()[1], (elmo_embeddings.dim(), len(sentence))
        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()
        embeddings = []
        cur_word_index = 0
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and (
                    (np.random.rand() < 1 / (1 + count)) or (np.random.rand() < 0.2))):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            input_components = [word_embedding]
            if self.use_elmo:
                if tag == START or tag == STOP:
                    elmo_embedding = dy.zeros(1024)
                else:
                    elmo_weights = dy.parameter(self.elmo_weights)
                    elmo_embedding = dy.sum_dim(dy.cmult(elmo_weights, dy.pick(elmo_embeddings,
                                                                               index=cur_word_index,
                                                                               dim=1)), [0])
                    cur_word_index += 1
                input_components.append(elmo_embedding)
            if not self.predict_pos:
                tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
                input_components.append(tag_embedding)

            raw_input = dy.concatenate(input_components)
            # if is_train:
            #     input = dy.dropout(raw_input, p=0.1)
            # else:
            input = raw_input
            embeddings.append(input)
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

    def get_distribution_for_kbest(self, sentence, elmo_embeddings, cur_word_index):
        assert self.empty_label_index == 0, self.empty_label_index
        lstm_outputs = self._featurize_sentence(sentence, is_train=False,
                                                elmo_embeddings=elmo_embeddings)
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
        label_log_probabilities_np = np.log(10 ** -6 + np.array(
            [non_constituent_probabilities, 1 - non_constituent_probabilities]))
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

    def compute_label_distributions(self, sentence, is_train, elmo_embeddings, cur_word_index):
        lstm_outputs = self._featurize_sentence(sentence, is_train=is_train,
                                                elmo_embeddings=elmo_embeddings)
        encodings = []
        span_to_index = {}
        for start in range(0, len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span_to_index[(start, end)] = len(encodings)
                encodings.append(self._get_span_encoding(start, end, lstm_outputs))
        label_log_probabilities = self._encodings_to_label_log_probabilities(encodings)
        return span_to_index, label_log_probabilities

    def _span_parser_predict_pos(self, sentence, is_train, elmo_embeddings, gold=None):
        if gold is not None:
            assert isinstance(gold, ParseNode)

        if is_train:
            regularization_on = random.random() > 0.1
        else:
            regularization_on = False

        lstm_outputs = self._featurize_sentence(sentence, is_train=regularization_on,
                                                elmo_embeddings=elmo_embeddings)

        other_encodings = []
        single_word_encodings = []
        temporary_span_to_index = {}
        for left in range(len(sentence)):
            for right in range(left + 1, len(sentence) + 1):
                encoding = self._get_span_encoding(left, right, lstm_outputs)
                span = (left, right)
                if right - left == 1:
                    temporary_span_to_index[span] = len(single_word_encodings)
                    single_word_encodings.append(encoding)
                else:
                    temporary_span_to_index[span] = len(other_encodings)
                    other_encodings.append(encoding)

        encodings = single_word_encodings + other_encodings
        span_to_index = {}
        for span, index in temporary_span_to_index.items():
            if span[1] - span[0] == 1:
                new_index = index
            else:
                new_index = index + len(single_word_encodings)
            span_to_index[span] = new_index
        span_encodings = dy.rectify(dy.reshape(self.f_encoding(dy.concatenate_to_batch(encodings)),
                                               (self.hidden_dim, len(encodings))))
        label_scores = self.f_label(span_encodings)
        label_scores_reshaped = dy.reshape(label_scores, (self.label_vocab.size, len(encodings)))
        label_log_probabilities = dy.log_softmax(label_scores_reshaped)
        single_word_span_encodings = dy.select_cols(span_encodings,
                                                    list(range(len(single_word_encodings))))
        tag_scores = self.f_tag(single_word_span_encodings)
        tag_scores_reshaped = dy.reshape(tag_scores,
                                         (self.tag_vocab.size, len(single_word_encodings)))
        tag_log_probabilities = dy.log_softmax(tag_scores_reshaped)

        if is_train:
            total_loss = dy.zeros(1)
            span_to_gold_label = get_all_spans(gold)
            for span, oracle_label in span_to_gold_label.items():
                oracle_label_index = self.label_vocab.index(oracle_label)
                index = span_to_index[span]
                if span[1] - span[0] == 1:
                    oracle_tag = sentence[span[0]][0]
                    total_loss -= tag_log_probabilities[self.tag_vocab.index(oracle_tag)][index]
                total_loss -= label_log_probabilities[oracle_label_index][index]
            return total_loss
        else:
            label_log_probabilities_np = label_log_probabilities.npvalue()
            tag_log_probabilities_np = tag_log_probabilities.npvalue()
            sentence_with_tags = []
            num_correct = 0
            total = 0
            # print('output has gold pos tags')
            for word_index, (oracle_tag, word) in enumerate(sentence):
                tag_index = np.argmax(tag_log_probabilities_np[:, word_index])
                tag = self.tag_vocab.value(tag_index)
                oracle_tag_is_deletable = oracle_tag in deletable_tags
                predicted_tag_is_deletable = tag in deletable_tags
                if oracle_tag is not None:
                    oracle_tag_index = self.tag_vocab.index(oracle_tag)
                    if oracle_tag_index == tag_index and tag != oracle_tag:
                        if oracle_tag[0] != '-':
                            print(tag, oracle_tag)
                        tag = oracle_tag
                    num_correct += tag_index == oracle_tag_index

                if oracle_tag is not None and oracle_tag_is_deletable != predicted_tag_is_deletable:
                    # print('falling back on gold tag', oracle_tag, tag)
                    sentence_with_tags.append((oracle_tag, word))
                else:
                    sentence_with_tags.append((tag, word))

                total += 1
            tree, additional_info = optimal_parser(label_log_probabilities_np,
                                                   span_to_index,
                                                   sentence_with_tags,
                                                   self.empty_label_index,
                                                   self.label_vocab,
                                                   gold)
            return tree, (additional_info, num_correct, total)

    def _span_parser_given_pos(self, sentence, is_train, elmo_embeddings, gold=None):
        print('Using Given POS')
        if gold is not None:
            assert isinstance(gold, ParseNode)

        lstm_outputs = self._featurize_sentence(sentence, is_train=is_train,
                                                elmo_embeddings=elmo_embeddings)

        encodings = []
        span_to_index = {}
        for left in range(len(sentence)):
            for right in range(left + 1, len(sentence) + 1):
                encoding = self._get_span_encoding(left, right, lstm_outputs)
                span_to_index[(left, right)] = len(encodings)
                encodings.append(encoding)

        label_scores = self.f_label(dy.concatenate_to_batch(encodings))
        label_scores_reshaped = dy.reshape(label_scores, (self.label_vocab.size, len(encodings)))
        label_log_probabilities = dy.log_softmax(label_scores_reshaped)

        if is_train:
            total_loss = dy.zeros(1)
            span_to_gold_label = get_all_spans(gold)
            for span, oracle_label in span_to_gold_label.items():
                oracle_label_index = self.label_vocab.index(oracle_label)
                index = span_to_index[span]
                total_loss -= label_log_probabilities[oracle_label_index][index]
            return total_loss
        else:
            label_log_probabilities_np = label_log_probabilities.npvalue()
            num_correct = 0
            total = 0
            tree, additional_info = optimal_parser(label_log_probabilities_np,
                                                   span_to_index,
                                                   sentence,
                                                   self.empty_label_index,
                                                   self.label_vocab,
                                                   gold)
            return tree, (additional_info, num_correct, total)

    def span_parser(self, sentence, is_train, elmo_embeddings, gold=None):
        if self.predict_pos:
            return self._span_parser_predict_pos(sentence, is_train, elmo_embeddings, gold)
        else:
            raise Exception('unimplemented')
            # return self._span_parser_given_pos(sentence, is_train, elmo_embeddings, gold)

    def fine_tune_confidence(self, sentence, lmbd, alpha, elmo_embeddings, cur_word_index, gold):
        lstm_outputs = self._featurize_sentence(sentence, is_train=False,
                                                elmo_embeddings=elmo_embeddings)
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
