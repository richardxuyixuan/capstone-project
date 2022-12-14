# capstone-project

**To train:**
1. change the codes for L9-11 in train_pipeline.py
2. (optionally) change other args on top of train_pipeline.py
3. (optionally) add a new model in models.py and use that instead
4. Run train_pipeline.py

**To test:**
1. change the codes for L16-18 in eval_pipeline.py
2. (optionally) change other args on top of eval_pipeline.py
3. Run eval_pipeline.py

**Try to improve your models (or at least run both train and eval):**
1. Wendy (caption only)
2. Skyler (image only)
3. Richard (early and late fusion)

**Some strategies to consider:**
1. Try the segmented images  on CNN

**BERT related references:**
1. ```captions_to_BERT.py```: Convert captions to BERT embeddings (of dimension 768);
2. ```dim_reduct.py```: Add and train a PCA layer for the model to reduce the output dimension to 128;
3. ```reduced_BERT.py```: Convert captions to reduced BERT embeddings (of dimension 128), using the model trained by ```dim_reduct.py```.

**CLIP related references:**
1. ```img_emb.py```: Convert ads images to CLIP embeddings (of dimension 512);
2. ```finetune_CLIP.py```: Finetune clip models using caption-ads pairs in the given dataset and save the model. 

**Other Things to Try**

Might need to retrain early and late fusion without PCA

Attention

Remove learning scheduler

Remove augmented data



1. Attention mechanism
2. Focal loss and other losses
3. Weight initialization
4. Add industry as a feature for ad
5. User similarity score
6. Item (ad) similarity score
7. Data augmentation 
8. Different combination of features
9. Ordinal logistic regression
10. Fix confusion matrix
11. P@K with k=number of ads with score 5
12. DEMO and trying the model on new dataset that we create ourselves
13. Five classes classification
14. Serial coeffiicents
15. Binary → Binary → binary
16. Leranable thresholds
17. Add regression score back
