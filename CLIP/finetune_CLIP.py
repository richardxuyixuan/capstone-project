from docarray import Document, DocumentArray
import os
import pandas as pd
import finetuner
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

bcaption_df = pd.read_csv('ads_captions_1027.csv', header=None)
bcaption = dict(bcaption_df.itertuples(False, None))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('clip-ViT-B-32')

def preprocess_and_encode_image(d: Document):
    """Preprocess image and extract embeddings from CLIP image encoder"""
    d.tensor = Image.open(d.uri)
    d.embedding = model.encode(d.tensor)
    d.pop('tensor')
    return d

def assign_labels(d: Document):
    filename = d.uri.split('\\')[1]

    d.tags['finetuner_label'] = bcaption[filename]
    return d

if __name__=="__main__":

    da = DocumentArray.from_files('./images/segmentation_result/*.*')

    da.apply(assign_labels, show_progress=True)
    da.apply(preprocess_and_encode_image, show_progress=True)

    images = []
    for i in range(1, 21, 1):
        for j in range(1, 16, 1):
            name = str(i) + "_" + str(j) + ".png"
            images = images + [name]

    directory = './images/segmentation_result'
    doc_arr = DocumentArray()

    for doc in da:
        pair = Document()
        img_chunk = doc.load_uri_to_image_tensor(224, 224)
        img_chunk.modality = 'image'
        txt_chunk = Document(content=doc.tags['finetuner_label'])
        txt_chunk.modality = 'text'
        pair.chunks.extend([img_chunk, txt_chunk])
        doc_arr.append(pair)
    # fine-tuning
    finetuner.login()

    run = finetuner.fit(
        model='openai/clip-vit-base-patch32',
        run_name='finetune-clip',
        train_data=doc_arr,
        loss='CLIPLoss',
    )

    #model = finetuner.get_model('openai/clip-vit-base-patch32')
    #finetuner.encode(model=model, data=doc_arr)

    artifact = run.save_artifact('clip-model')