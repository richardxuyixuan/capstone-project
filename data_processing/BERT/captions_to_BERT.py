import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel
import gc

def BERT_computation(tokens, left, right, bmodel):
    vecs = []
    for i in range(left, right, 1):
        print("Processing image # " + str(i))
        #token = tokens[i][None, :]
        token = tokens[i]
        caption_vec = bmodel(token).pooler_output.squeeze(0)
        vecs = vecs + [caption_vec]
    print(len(vecs))
    print(vecs[1].shape)
    return vecs

if __name__=="__main__":
    # Load all the captions
    bcaption_df = pd.read_csv('ads_captions_1027.csv', header=None)
    bcaption = dict(bcaption_df.itertuples(False, None))

    images = []
    for i in range(1, 21, 1):
        for j in range(1, 16, 1):
            name = str(i) + "_" + str(j) + ".png"
            images = images + [name]

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Load the model
    model = BertModel.from_pretrained('bert-base-uncased')

    # Determine the max sentence length
    max_len = 0

    for filename in images: #os.listdir(directory):
        # Note: 1_16.png is thrown out because no labels
        if filename == '1_16.png':
            continue
        cap = bcaption[filename]
        ids = tokenizer.encode(cap, add_special_tokens=True)
        max_len = max(max_len, len(ids))

    # Tokenization
    print("Computing BERT tokenization...")
    cnt = 0
    captions_token = []
    for filename in images:  # os.listdir(directory):
        # Note: 1_16.png is thrown out because no labels
        if filename == '1_16.png':
            continue
        print("Processing image # " + str(cnt) + " : " + filename)
        cap = bcaption[filename]
        encode_dict = tokenizer.encode_plus(cap, add_special_tokens=True, max_length=max_len,
                                            padding='max_length',
                                            return_attention_mask=True, return_tensors='pt')
        captions_token.append(encode_dict['input_ids'])
        cnt += 1

    # Compute BERT vectors (take CLS token)
    # captions_token = torch.cat(captions_token, dim=0)
    #captions_token = captions_token.to('cuda')
    #model.to('cuda')
    print("Computing BERT vectors...")
    #captions_vec = model(captions_token).pooler_output
    captions_vec = []
    # Break batches so that it does not kill my computer
    captions_vec = captions_vec + BERT_computation(captions_token, 0, 150, model)
    captions_vec = captions_vec + BERT_computation(captions_token, 150, len(captions_token), model)
    captions_vec = torch.stack(captions_vec)
    #captions_vec = torch.cat(captions_vec, dim=0)
    '''
    cnt = 0
    for i in range(0, captions_token.shape[0], 1):
        print("Processing image # " + str(cnt) + " : " + str(images[cnt]))
        token = captions_token[i][None, :]
        #token.to('cuda')
        caption_vec = model(token).pooler_output.squeeze(0)
        captions_vec = captions_vec + [caption_vec]
        cnt += 1
    '''

    # Check results:
    print("Computation is completed! The output has the following type and size:")
    print(captions_vec.type)
    print(captions_vec.shape)

    # Store all the embeddings as a pickle data file
    with open('./caption_bert_768.data', 'wb') as filehandle:
        pickle.dump(captions_vec, filehandle)