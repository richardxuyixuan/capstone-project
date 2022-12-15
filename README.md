# Leveraging Vision-Language Fusion in Ad Recommendation System

## Capstone Project for AdView

### Yixuan Xu, Sumin Lee, Jianzhong (Ken) Shi, Yutong Wang, Jiarui Zhang

**First Time Use**

Please ensure that you have the following dependencies.

```
pytorch 1.13.0
numpy 1.23.3
sklearn 1.1.3
seaborn 0.12.1
pandas 1.5.1
```


**Model Evaluation**

1. To use the best model, please download the checkpoint from [here](https://drive.google.com/file/d/1ruh2ktmOB24L3emESONHkfbiDu3P6rSl/view?usp=sharing). Alternatively, you can use evaluate any other experiments that you have trained on. See note 1 below on how to do so.
2. Run ```eval_pipeline.py```

Note 1: The arguments provided in ```eval_pipeline.py``` are by default the setting for the best model checkpoint. However, feel free to change it. For example, if you performed any training, you may wish to update the hyperparameters do match the training hyperparameter settings.

**Model Training**

1. For the final model training setting, please use the default settings. If you would like to change the hyperparameter settings, you may do so as well. See note 2 below on how to do so.
2. Run```train_pipeline.py```

Note 2: The arguments provided in ```train_pipeline.py``` are by default the setting for the best model checkpoint. However, feel free to change it. For example, you may wish to set the output_dim to 5 to see how the model performs for classification performance on 5 classes.

**Processing New Data: Generating a Caption for a New Advertisement Image**
1. Download the following checkpoints [here](https://drive.google.com/file/d/1tKHj7DSDtOUBUGrLU1zkNWEE1AbLZT5Z/view?usp=sharing) and [here](https://drive.google.com/file/d/1H-sY3C6q72a4eu4tAGpqi7oHtmd3irwX/view?usp=sharing) and place them on the same directory as ```data_processing.py```
2. ```data_processing.py```, run img_process function. This function requires four inputs. input_image_path is the path to ad image, output_image_path is the path to processed ad image.resnet_weights_path and classifier_weights_path are the paths to saved pretrained models weights. img_process function will read the ad image from input_image_path and write a processed image to output_image_path. 
3. Download the GenerativeImage2Text repo from [Github](https://github.com/microsoft/GenerativeImage2Text). Follow their README file to perform inference on a single image. Here's an example of the command line:
```
AZFUSE_TSV_USE_FUSE=1 python3 -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', 
      'image_path': './GenerativeImage2Text/aux_data/images', \
      'model_name': 'GIT_LARGE_TEXTCAPS', \
      'prefix': '', \
      'result_file': './GenerativeImage2Text'\
}"
```
The model name should be GIT_LARGE_TEXTCAPS, the image should be placed inside aux_data/images under the GenerativeImage2Text folder. Then a txt file with the generated capstion will be generated under the 'result_file' folder. 

**Processing New Data: Training Classifier in Class-aware Image Segmentation Tool from Scratch**
1. ```classifier.py```, run train() function. This function requires five inputs. annotation_file is the path to a csv file which contains the ground truth label for each advertisement image. img_dir is the path to a folder which contains all advertisement images (incluidng train, val, test). Other inputs are the hyperparameter settings. Feel free to change it. 


**Procsesing New Data: Performing Data Augmentation and Saving Augmented Data**
1. ```data_augmentation.py``` and run augmentation_img() function. This function requires a input path to read the image and a output path to write the augmented image. The function will apply a blur filter and increase brightness to the image. 

**Processing New Data: Generating BERT Caption Embedding**
1. ```captions_to_BERT.py```: Convert captions to BERT embeddings (of dimension 768);
2. ```dim_reduct.py```: Add and train a PCA layer for the model to reduce the output dimension to 128;
3. ```reduced_BERT.py```: Convert captions to reduced BERT embeddings (of dimension 128), using the model trained by ```dim_reduct.py```.

**Processing New Data: Generating CLIP Image Embedding**
1. ```img_emb.py```: Convert ads images to CLIP embeddings (of dimension 512);
2. ```finetune_CLIP.py```: Finetune clip models using caption-ads pairs in the given dataset and save the model. 
