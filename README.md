# Leveraging Vision-Language Fusion in Ad Recommendation System

## Capstone Project for AdView

### Yixuan Xu, Sumin Lee, Jianzhong (Ken) Shi, Yutong Wang, Jiarui Zhang


**Model Evaluation**

1. To use the best model, please download the checkpoint from 
2. Run ```python eval_pipeline.py```

The arguments provided in ```eval_pipeline.py``` are by default the setting for the best model checkpoint. However, feel free to change it. For example, if you performed any training, you may wish to update the hyperparameters do match the training hyperparameter settings.

**Model Training**

Run ```python train_pipeline.py```

The arguments provided in ```train_pipeline.py``` are by default the setting for the best model checkpoint. However, feel free to change it. For example, you may wish to set the output_dim to 5 to see how the model performs for classification performance on 5 classes.


**Try to improve your models (or at least run both train and eval):**
1. Wendy (caption only)
2. Skyler (image only)
3. Richard (early and late fusion)

**Some strategies to consider:**
1. Try the segmented images  on CNN


**How to generate a caption for a new advertisement image:**
1. open data_processing.py, run img_process function. This function requires four inputs. input_image_path is the path to ad image, output_image_path is the path to processed ad image.resnet_weights_path and classifier_weights_path are the paths to saved pretrained models weights. img_process function will read the ad image from input_image_path and write a processed image to output_image_path. 
2. Download the GenerativeImage2Text repo. Follow their README file to inference on a single image. Here's an example of the command line:
AZFUSE_TSV_USE_FUSE=1 python3 -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', 
      'image_path': './GenerativeImage2Text/aux_data/images', \
      'model_name': 'GIT_LARGE_TEXTCAPS', \
      'prefix': '', \
      'result_file': './GenerativeImage2Text'\
}"
The model name should be GIT_LARGE_TEXTCAPS, the image should be placed inside aux_data/images under the GenerativeImage2Text folder. Then a txt file with the generated capstion will be generated under the 'result_file' folder. 

**How to train a new classifier:**
1. open classifier.py, run train() function. This function requires five inputs. annotation_file is the path to a csv file which contains the ground truth label for each advertisement image. img_dir is the path to a folder which contains all advertisement images (incluidng train, val, test). Other inputs are the hyperparameter settings. Feel free to change it. 


**How to augment data:**
1. open data_augmentation.py and run augmentation_img() function. This function requires a input path to read the image and a output path to write the augmented image. The function will apply a blur filter and increase brightness to the image. 

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
