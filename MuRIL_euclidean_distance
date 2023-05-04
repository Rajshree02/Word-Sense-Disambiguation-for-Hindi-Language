# prerequisites
pip install bert-for-tf2
pip install tensorflow-text
!pip install tensorflow-datasets

# import statements
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_text as text
from bert import bert_tokenization
import numpy as np
from scipy.spatial import distance


#model

def get_model(model_url, max_seq_length):
    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
    )

    muril_layer = hub.KerasLayer(model_url, trainable=True)
    outputs = muril_layer(inputs)

    assert 'sequence_output' in outputs
    assert 'pooled_output' in outputs
    assert 'encoder_outputs' in outputs
    assert 'default' in outputs
    return tf.keras.Model(inputs=inputs,outputs=outputs["pooled_output"]), muril_layer
     

max_seq_length = 128
muril_model, muril_layer = get_model(
model_url="https://tfhub.dev/google/MuRIL/1", max_seq_length=max_seq_length)
     

vocab_file = muril_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = muril_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


def create_input(input_strings, tokenizer, max_seq_length):
    input_ids_all, input_mask_all, input_type_ids_all = [], [], []
    for input_string in input_strings:
        input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        sequence_length = min(len(input_ids), max_seq_length)
    
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length]
        else:
            input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

        input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

        input_ids_all.append(input_ids)
        input_mask_all.append(input_mask)
        input_type_ids_all.append([0] * max_seq_length)
  
    return np.array(input_ids_all), np.array(input_mask_all), np.array(input_type_ids_all)
     

def encode(input_text):
    input_ids, input_mask, input_type_ids = create_input(input_text, 
                                                       tokenizer, 
                                                       max_seq_length)
    inputs = dict(
        input_word_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids,
    )
    return muril_model(inputs)
    
    
# Run below code 
"""
sentences = ["खेल", "पेड़","पत्ते","नीचे","बैठना"]
embeddings = encode(sentences)

     
dst_1 = distance.euclidean(np.array(embeddings[0]), 
                           np.array(embeddings[1]))
print("Distance between {} & {} is {}".format(sentences[0],
                                                sentences[1],
                                                dst_1))
"""
