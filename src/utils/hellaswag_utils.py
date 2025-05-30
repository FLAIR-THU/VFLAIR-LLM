import collections
import csv
import os

def bert_pad(tokens_a, tokens_b, max_seq_length, tokenizer, do_mask=True, max_predictions_per_seq=10):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    if do_mask:
        # Create the masked tokens
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
            tokens,
            # masked_lm_prob=.15,
            masked_lm_prob=.05,
            max_predictions_per_seq=max_predictions_per_seq,
            vocab_words=tokenizer.vocab_words)

        masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        # Zero pad
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)
    else:
        masked_lm_positions, masked_lm_ids, masked_lm_weights = None, None, None

    # NOW do ittt
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, tokens


def convert_single_example(ex_index, example, max_seq_length, tokenizer, label_length=1, do_mask=True, max_predictions_per_seq=10):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    # # This will only be for testing so OK
    # if isinstance(example, PaddingInputExample):
    #     return InputFeatures(
    #         input_ids=[[0] * max_seq_length] * label_length,
    #         input_mask=[[0] * max_seq_length] * label_length,
    #         segment_ids=[[0] * max_seq_length] * label_length,
    #         label_id=0,
    #         masked_lm_positions=[[0] * max_predictions_per_seq] * label_length,
    #         masked_lm_ids=[[0] * max_predictions_per_seq] * label_length,
    #         masked_lm_weights=[[0.0] * max_predictions_per_seq] * label_length,
    #         is_real_example=False)

    tokens_bs = [tokenizer.tokenize(tb) for tb in example.text_b]
    if example.text_a:
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_as = [[x for x in tokens_a] for i in range(len(tokens_bs))]

        for i in range(len(tokens_bs)):
            _truncate_seq_pair(tokens_as[i], tokens_bs[i], max_seq_length - 3)

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, tokens = zip(
            *[bert_pad(token_a, token_b, max_seq_length, tokenizer, do_mask=do_mask,
                       max_predictions_per_seq=max_predictions_per_seq)
              for token_a, token_b in zip(tokens_as, tokens_bs)])

    else:
        tokens_bs = [x[-(max_seq_length - 2):] for x in tokens_bs]

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, tokens = zip(
            *[bert_pad(token_b, None, max_seq_length, tokenizer, do_mask=do_mask,
                       max_predictions_per_seq=max_predictions_per_seq)
              for token_b in tokens_bs])
    if not do_mask:
        masked_lm_positions, masked_lm_ids, masked_lm_weights = None, None, None

    if ex_index < 5:
        print("*** Example ***")
        print("guid:", example.guid)
        print("label:", example.label)

        for i, these_tokens in enumerate(tokens):
            print("Ending: {} / {}".format(i, len(tokens)))
            print("tokens: {}".format(' '.join([tokenization.printable_text(x) for x in these_tokens])))
            print("input_ids:", " ".join([str(x) for x in input_ids[i]]))
            print("input_mask:"," ".join([str(x) for x in input_mask[i]]))
            print("segment_ids"." ".join([str(x) for x in segment_ids[i]]))
            if do_mask:
                print("masked_lm_positions: ", " ".join([str(x) for x in masked_lm_positions[i]]))
                print("masked_lm_ids: ", " ".join([str(x) for x in masked_lm_ids[i]]))
                print("masked_lm_weights: ", " ".join([str(x) for x in masked_lm_weights[i]]))

    assert len(input_ids) == label_length

    feature = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label,
        masked_lm_positions=masked_lm_positions,
        masked_lm_ids=masked_lm_ids,
        masked_lm_weights=masked_lm_weights,
        is_real_example=True)
    return feature

def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file, label_length=1, do_mask=True, max_predictions_per_seq=10):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, label_length=label_length,
                                         do_mask=do_mask,
                                         max_predictions_per_seq=max_predictions_per_seq)

        # Flatten here
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        # We will have as input
        # [a0 a1 a2 a3]
        # [b0 b1 b2 b3]
        # seq_length is the last dimension
        def create_int_feature_flat(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=[v for x in values for v in x]))
            return f

        def create_float_feature_flat(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=[v for x in values for v in x]))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature_flat(feature.input_ids)
        features["input_mask"] = create_int_feature_flat(feature.input_mask)
        features["segment_ids"] = create_int_feature_flat(feature.segment_ids)
        if do_mask:
            features["masked_lm_positions"] = create_int_feature_flat(feature.masked_lm_positions)
            features["masked_lm_ids"] = create_int_feature_flat(feature.masked_lm_ids)
            features['masked_lm_weights'] = create_float_feature_flat(feature.masked_lm_weights)

        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks. OR if it's a list, you have N of these.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def _part_a(item):
    # if FLAGS.endingonly:
    #     return ''
    if 'ctx_a' not in item:
        return item['ctx']
    if 'ctx' not in item:
        return item['ctx_a']
    if len(item['ctx']) == len(item['ctx_a']):
        return item['ctx']
    return item['ctx_a']

def _part_bs(item):
    if ('ctx_b' not in item) or len(item['ctx_b']) == 0:
        return item['endings']
    return ['{} {}'.format(item['ctx_b'], x) for x in item['endings']]
