from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import torch

if __name__=="__main__":
    bcaption_df = pd.read_csv('ads_captions_1027.csv', header=None)
    bcaption = dict(bcaption_df.itertuples(False, None))

    caption_df = pd.read_csv('captions_1109.txt')
    git_images = caption_df['image'].tolist()
    git_captions = caption_df['caption'].tolist()

    images = []
    for i in range(1, 21, 1):
        for j in range(1, 16, 1):
            name = str(i) + "_" + str(j) + ".png"
            images = images + [name]

    captions = []
    for filename in images:
        # Note: 1_16.png is thrown out because no labels
        if filename == '1_16.png':
            continue
        cap = bcaption[filename]
        captions = captions + [cap]

    gcaptions = []
    for filename in images:
        # Note: 1_16.png is thrown out because no labels
        if filename == '1_16.png':
            continue
        cap = git_captions[git_images.index(filename)]
        gcaptions = gcaptions + [cap]

    with open('most_sentence.txt') as file:
        most = [line.rstrip() for line in file]
    with open('user_sentence.txt') as file:
        users = [line.rstrip() for line in file]

    model = SentenceTransformer('models/my-128dim-model')

    captions_vec = model.encode(captions)
    np.stack(captions_vec, axis=0)
    captions_vec = torch.from_numpy(captions_vec)

    gcaptions_vec = model.encode(gcaptions)
    np.stack(gcaptions_vec, axis=0)
    gcaptions_vec = torch.from_numpy(gcaptions_vec)

    most_vec = model.encode(most)
    np.stack(most_vec, axis=0)
    most_vec = torch.from_numpy(most_vec)

    users_vec = model.encode(users)
    np.stack(users_vec, axis=0)
    users_vec = torch.from_numpy(users_vec)

    with open('./caption_bert_128.data', 'wb') as filehandle:
        pickle.dump(captions_vec, filehandle)
    with open('./caption_1109_bert_128.data', 'wb') as filehandle:
        pickle.dump(gcaptions_vec, filehandle)
    with open('./most_bert_128.data', 'wb') as filehandle:
        pickle.dump(most_vec, filehandle)
    with open('./user_bert_128.data', 'wb') as filehandle:
        pickle.dump(users_vec, filehandle)