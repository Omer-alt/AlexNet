# AlexNet paper implementation on ImageNet

![AlexNet](./assets/AlexNetarchitecture.png)
Network architecture of AlexNet.

In this experiment, we aimed to implement and evaluate the AlexNet architecture on the ImageNet dataset, focusing on various aspects such as data preprocessing, augmentation, and model performance.

### 1. Data Loading


### 2. Data Loading
For the validation dataset, we mapped each image to its corresponding class using XML annotations. This involved extracting image names, parsing XML files to obtain class labels, and saving the results in a CSV file. Additionally, we created a class mapping dictionary from a provided mapping file to facilitate the association of images with their respective classes. A custom dataset class was implemented to load images and labels from directories, with an option to limit the dataset size for quicker experiments. Transformations were applied to standardize and prepare the data for training.






**Result:** 
![Run_Over_Epochs_7](./assets/Epochs.png)
Network architecture of AlexNet.
The result on the complete dataset ILSVRC- 2012 run on Kaggle during 12 hours of time




### Build With

**Language:** Python

**Package:** python-dotenv, seaborn, torchvision, matplotlib, Pytorch

### Run Locally

Clone the project
```bash
    git clone https://github.com/Omer-alt/AlexNet.git
```

Go to the project directory and run it.
```bash
    cd AlexNet
    python3 main.py
```
### License

[MIT](https://choosealicense.com/licenses/mit/)