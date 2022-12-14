from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pandas as pd
import os
import pickle
import torch

if __name__=="__main__":

    # Load CLIP model
    model = SentenceTransformer('clip-ViT-B-32')

    # Load file names
    caption_df = pd.read_csv('caption.txt')
    mcaption = dict(caption_df.itertuples(False, None))
    images = []
    for i in range(1, 21, 1):
        for j in range(1, 16, 1):
            name = str(i) + "_" + str(j) + ".png"
            images = images + [name]
    #print(images)
    #images = caption_df['image'].tolist()
    #print(images)

    # Load manual captions
    bcaption_df = pd.read_csv('ads_captions_1027.csv', header=None)
    bcaption = dict(bcaption_df.itertuples(False, None))
    #print(bcaption)

    # Generate Image embeddings for all segmented images
    img_embs = []
    sim_scores = []

    directory = './images/segmentation_result'
    for filename in images: #os.listdir(directory):
        # Note: 1_16.png is thrown out because no labels
        if filename == '1_16.png':
            continue
        print("Processing " + filename + " ...")
        f = os.path.join(directory, filename)
        img_emb = model.encode(Image.open(f))
        img_embs = img_embs + [torch.Tensor(img_emb)]
        cap = bcaption[filename]
        #cap = mcaption[filename]
        cap_emb = model.encode(cap)
        cos_score = util.cos_sim(img_emb, cap_emb)
        print(cos_score.item())
        sim_scores = sim_scores + [cos_score.item()]

    img_embs = torch.stack(img_embs)

    # Check results:
    print("Computation is completed! The output has the following type and size:")
    print(img_embs.type)
    print(img_embs.shape)
    print("The average similarity score is:")
    print(sum(sim_scores) / len(sim_scores))

    # Store all the embeddings and scores as pickle data files
    with open('./img_embs_base_512.data', 'wb') as filehandle:
        pickle.dump(img_embs, filehandle)
    with open('./sim_scores_base_512.data', 'wb') as filehandle:
        pickle.dump(sim_scores, filehandle)