import pandas as pd
import numpy as np
import tensorflow as tf
import time, sys
import re
import pickle
import argparse
import os
from model import Transformer
from utils import *
from scheduler import CustomSchedule

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model():
    transformer = Transformer(
        opt.num_layers,
        opt.d_model,
        opt.num_heads,
        opt.dff,
        encoder_vocab_size,
        decoder_vocab_size,
        pe_input=encoder_vocab_size,
        pe_target=decoder_vocab_size,
    )
    return transformer


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return (tf.reduce_sum(loss_) / tf.reduce_sum(mask)) / num_gpus


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                     tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp,
            True,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_accuracy.update_state(tar_real, predictions)

    return loss


@tf.function
def distributed_train_step(inp_dis, tar_dis):
    per_replica_losses = strategy.run(train_step, args=(inp_dis, tar_dis, ))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder_max_len', type=int, default=2000)
    parser.add_argument('-decoder_max_len', type=int, default=216)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_layers', type=int, default=4)
    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-dff', type=int, default=2048)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-encoder_max_vocab', type=int, default=100000)
    parser.add_argument('-decoder_max_vocab', type=int, default=100000)
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-data_path', type=str, required=True)
    parser.add_argument('-checkpoint_path', type=str, required=True)
    parser.add_argument('-vocab_save_dir', type=str, required=True)
    parser.add_argument('-filters', action='store_true')
    parser.add_argument('-no_filters', action='store_true')

    opt = parser.parse_args()

    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    print('### Number of devices: {} ...'.format(num_gpus))

    if not os.path.exists(opt.vocab_save_dir):
        os.makedirs(opt.vocab_save_dir)

    assert(opt.filters or opt.no_filters)
    assert(not(opt.filters and opt.no_filters))
    if opt.filters:
        filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
        print('filters = [{}]'.format(filters))
    if opt.no_filters:
        filters = '\t\n'
        print('filters = [{}]'.format(filters))

    oov_token = '<unk>'

    news = pd.read_excel(opt.data_path, dtype=str)
    news.drop(['id_articles'], axis=1, inplace=True)

    documents = news['articles']
    summaries = news['abstracts']
    summaries = summaries.apply(lambda x: '<go> ' + x + ' <stop>')

    print("### Tokenizing the texts into integer tokens...", file=sys.stderr)
    if opt.encoder_max_vocab != -1:
        document_tokenizer = tf.keras.preprocessing.text.Tokenizer(
          num_words=opt.encoder_max_vocab, filters=filters,
          oov_token=oov_token)
    else:
        document_tokenizer = tf.keras.preprocessing.text.Tokenizer(
          filters=filters, oov_token=oov_token)

    if opt.decoder_max_vocab != -1:
        summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(
          num_words=opt.decoder_max_vocab, filters=filters,
          oov_token=oov_token)
    else:
        summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(
          filters=filters, oov_token=oov_token)

    document_tokenizer.fit_on_texts(documents)
    summary_tokenizer.fit_on_texts(summaries)

    with open(
      os.path.join(opt.vocab_save_dir,
                   f'document_tokenizer_{opt.encoder_max_vocab}.pickle'),
      'wb') as fp:
        pickle.dump(document_tokenizer, fp)

    with open(
      os.path.join(opt.vocab_save_dir,
                   f'summary_tokenizer_{opt.decoder_max_vocab}.pickle'),
      'wb') as fp:
        pickle.dump(summary_tokenizer, fp)

    inputs = document_tokenizer.texts_to_sequences(documents)
    targets = summary_tokenizer.texts_to_sequences(summaries)

    if opt.encoder_max_vocab != -1:
        encoder_vocab_size = opt.encoder_max_vocab
    else:
        encoder_vocab_size = len(document_tokenizer.word_index) + 1

    if opt.decoder_max_vocab != -1:
        decoder_vocab_size = opt.decoder_max_vocab
    else:
        decoder_vocab_size = len(summary_tokenizer.word_index) + 1

    print("### Obtaining insights on lengths for defining maxlen...",
          file=sys.stderr)
    document_lengths = pd.Series([len(x) for x in documents])
    summary_lengths = pd.Series([len(x) for x in summaries])
    BUFFER_SIZE = int(document_lengths.count())

    print("### Padding/Truncating sequences for identical sequence lengths...",
          file=sys.stderr)
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
      inputs, maxlen=opt.encoder_max_len, padding='post', truncating='post')
    targets = tf.keras.preprocessing.sequence.pad_sequences(
      targets, maxlen=opt.decoder_max_len, padding='post', truncating='post')

    print("### Creating dataset pipeline...", file=sys.stderr)
    inputs = tf.cast(inputs, dtype=tf.int32)
    targets = tf.cast(targets, dtype=tf.int32)

    dataset_train = tf.data.Dataset.from_tensor_slices(
      (inputs, targets)).shuffle(BUFFER_SIZE).batch(opt.batch_size)

    with strategy.scope():
        print("### Creating model...", file=sys.stderr)
        transformer = create_model()
        print('### Defining losses and other metrics...', file=sys.stderr)
        learning_rate = CustomSchedule(opt.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                             beta_2=0.98, epsilon=1e-9)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction='none')

    with strategy.scope():
        train_loss = tf.keras.metrics.Mean(name='train_loss')

    print("### Setting checkpoints manager...", file=sys.stderr)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, opt.checkpoint_path,
                                              max_to_keep=5)

    with strategy.scope():
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          name='train_accuracy')

    print("### Training ...", file=sys.stderr)
    with strategy.scope():

        for epoch in range(opt.epochs):
            # start_time = time.time()
            total_loss = 0.0
            num_batches = len(dataset_train)
            train_accuracy.reset_states()

            for batch, (inp, tar) in enumerate(dataset_train):

                total_loss += distributed_train_step(inp, tar)

                if batch % 2 == 0:
                    print(f'Epoch {epoch + 1}/{opt.epochs} "
                          f"Batch {batch + 1}/{num_batches}',
                          file=sys.stderr)

            train_loss = total_loss / num_batches

            print(f"Epoch {epoch + 1}, Loss: {train_loss}, "
                  f"Accuracy: {train_accuracy.result()*100}",
                  file=sys.stderr)
            end_time = time.time()
            print(f"Time: {end_time - start_time}", file=sys.stderr)

            print('### Save the checkpoints ...', file=sys.stderr)
            if (epoch + 1) % 1 == 0:
                path_save_ckp = opt.checkpoint_path + 'epoch_' + str(epoch + 1)

                if not os.path.isdir(path_save_ckp):
                    os.makedirs(path_save_ckp)
                    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                              path_save_ckp,
                                                              max_to_keep=5)
                    ckpt_save_path = ckpt_manager.save()
                    print(f'Saving checkpoint for epoch {epoch + 1} '
                          f'at {ckpt_save_path}',
                          file=sys.stderr)
