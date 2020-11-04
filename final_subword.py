import tensorflow as tf
import numpy as np
import time
import pandas as pd
import logging
import os
import sentencepiece as spm
#tf.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from sklearn.model_selection import train_test_split


file1=open('english.txt','r',encoding='utf-8')
file2=open('gujarati.txt','r',encoding='utf-8')

t1=[]
for i in file1.readlines():
    t1.append(i[:-1])
t2=[]
for i in file2.readlines():
    t2.append(i[:-1])


raw_data = pd.DataFrame(list(zip(t1, t2)), columns =['eng', 'guj'])

# split data into train and test set
train, test = train_test_split(raw_data.values, test_size=0.3, random_state = 12)

# data preprocessing
raw_data_en=list(train[:,0])
raw_data_fr=list(train[:,1])

spm.SentencePieceTrainer.train('--input=gujarati.txt --model_prefix=g --vocab_size=2000')
sp = spm.SentencePieceProcessor()
sp.load('g.model')

spm.SentencePieceTrainer.train('--input=english.txt --model_prefix=e --vocab_size=2000')
sp1 = spm.SentencePieceProcessor()
sp1.load('e.model')


for i in range(len(raw_data_en)):
    raw_data_en[i]=" ".join(sp1.encode_as_pieces(raw_data_en[i]))
    raw_data_fr[i]=" ".join(sp.encode_as_pieces(raw_data_fr[i]))

raw_data_fr_in = ['<start> ' + data for data in raw_data_fr]
raw_data_fr_out = [data + ' <end>' for data in raw_data_fr]

test_en=list(test[:,0])

test_fr=list(test[:,1])

# tokenization
en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,padding='post')

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,padding='post')

BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices(
    (data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(20).batch(BATCH_SIZE)


# positional embedding
def positional_embedding(pos, model_size):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE

max_length = max(len(data_en[0]), len(data_fr_in[0]))
MODEL_SIZE = 512

pes = []
for i in range(max_length):
    pes.append(positional_embedding(i, MODEL_SIZE))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)


# Multi-head attention
class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))

    def call(self, decoder_output, encoder_output):
        # decoder_output has shape (batch, decoder_len, model_size)
        # encoder_output has shape (batch, encoder_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](decoder_output), self.wk[i](encoder_output), transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, decoder_len, encoder_len)
            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, decoder_len, encoder_len)
            head = tf.matmul(alignment, self.wv[i](encoder_output))
            # head has shape (batch, decoder_len, value_size)
            heads.append(head)
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        # heads has shape (batch, decoder_len, model_size)
        return heads
 

# Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size,embeddings_regularizer=tf.keras.regularizers.L2(l2=0.01))
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        
        self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        self.dense_1 = [tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)) for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
    def call(self, sequence):
        sub_in = []
        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            sub_in.append(embed + pes[i, :])
            
        sub_in = tf.concat(sub_in, axis=1)
        
        for i in range(self.num_layers):
            sub_out = []
            for j in range(sub_in.shape[1]):
                attention = self.attention[i](
                    tf.expand_dims(sub_in[:, j, :], axis=1), sub_in)

                sub_out.append(attention)

            sub_out = tf.concat(sub_out, axis=1)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)
            
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out
            
        return ffn_out

    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size,embeddings_regularizer=tf.keras.regularizers.L2(l2=0.01))
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        self.dense_1 = [tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)) for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        self.dense = tf.keras.layers.Dense(vocab_size, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))
        
    def call(self, sequence, encoder_output):
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = []
        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            embed_out.append(embed + pes[i, :])
            
        embed_out = tf.concat(embed_out, axis=1)
        
        
        bot_sub_in = embed_out
        
        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            bot_sub_out = []
            
            for j in range(bot_sub_in.shape[1]):
                values = bot_sub_in[:, :j, :]
                attention = self.attention_bot[i](
                    tf.expand_dims(bot_sub_in[:, j, :], axis=1), values)

                bot_sub_out.append(attention)
            bot_sub_out = tf.concat(bot_sub_out, axis=1)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)
            
            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = []
            for j in range(mid_sub_in.shape[1]):
                attention = self.attention_mid[i](
                    tf.expand_dims(mid_sub_in[:, j, :], axis=1), encoder_output)

                mid_sub_out.append(attention)

            mid_sub_out = tf.concat(mid_sub_out, axis=1)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out
        
        logits = self.dense(ffn_out)
            
        return logits
    
H = 2
NUM_LAYERS = 2

en_vocab_size = len(en_tokenizer.word_index) + 1
encoder = Encoder(en_vocab_size, MODEL_SIZE, NUM_LAYERS, H)


print('Input vocabulary size', en_vocab_size)

fr_vocab_size = len(fr_tokenizer.word_index) + 1
max_len_fr = data_fr_in.shape[1]
decoder = Decoder(fr_vocab_size, MODEL_SIZE, NUM_LAYERS, H)


print('Target vocabulary size', fr_vocab_size)



crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)
def loss_func(targets, logits):
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


optimizer = tf.keras.optimizers.Adam()

def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    #print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    #print(test_source_seq)

    en_output = encoder(tf.constant(test_source_seq))

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]], dtype=tf.int64)

    out_words = []

    while True:
        de_output = decoder(de_input, en_output)
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        out_words.append(fr_tokenizer.index_word[new_word.numpy()[0][0]])

        de_input = tf.concat((de_input, new_word), axis=-1)

        if out_words[-1] == '<end>' or len(out_words) >= 14:
            break

    return ' '.join(out_words)


@tf.function
def train_step(source_seq, target_seq_in, target_seq_out):
    with tf.GradientTape() as tape:
        encoder_output = encoder(source_seq)
        
        decoder_output = decoder(target_seq_in, encoder_output)

        loss = loss_func(target_seq_out, decoder_output)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in zip(gradients,variables)]
    optimizer.apply_gradients(capped_gvs)
    #optimizer.apply_gradients(zip(gradients, variables))

    return loss

NUM_EPOCHS = 1

start_time = time.time()
for e in range(NUM_EPOCHS):
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out)

    print('Epoch {} Loss {:.4f}'.format(
          e + 1, loss.numpy()))

end_time = time.time()
print('Average elapsed time: {:.2f}s'.format((end_time - start_time) / (e + 1)))

            
import nltk

bleu_sum=0
count=0
for i in range(len(test_en)):
    test_sequence=test_en[i]
    try:
        op=predict(test_sequence)
    except:
        count+=1
        continue
    op=sp.decode_pieces(op.split(' '))
    if i%1000==0:
        print(test_en[i])
        print(test_fr[i])
        print(op,'\n')
    BLEU = nltk.translate.bleu_score.sentence_bleu([test_fr[i]], op ,weights = (0.5, 0.5))
    bleu_sum+= BLEU

print("BLEU Score :",(bleu_sum/(len(test_en)-count))*100)