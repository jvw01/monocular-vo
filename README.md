# ReadMe for the Visual Odometry Pipeline in Vision Algorithms for Mobile Robotics Fall 2024

## Group members:
- Jakob Wolfram
- Clara Seuge
- Fabio Huebel
- Anton Pollak

## Screencasts:
These are the links to the screencasts recorded for each of the datasets:

- KITTI: [Kitti](https://youtu.be/iMiLZzel61M)
- Malaga: [Malaga](https://youtu.be/ksQePvSiVzQ)
- Parking: [Parking](https://youtu.be/6iXwq24SfQY)

## Specifications of the machine used to record the screencasts
The machine on which we recorded the screencasts was an ASUS Zenbook 14 OLED with the following specifications:
- Intel Evo i9 CPU
- 64 GB RAM
- NVidia GeForce RTX onboard graphics
- Runnning Ubuntu 22.04

The metrics while running the KITTI pipeline were the following, as seen in this screencast of the System Monitor taken while the pipeline was running:
[Kitti System Monitor](https://youtu.be/w8Ba_fKGgv0)


## Putting the datasets in the right place
We have zipped the entire running folder including the imagedataset and uploaded it to polybox (link in email).


In case you would like to place the datasets manually, we describe the required locations in the following:
From the base directory of this repository, the images of the datasets should be inside the following folder structure:

### KITTI:
```bash
data/kitti/05/image_0/
```
### Malaga:
```bash
data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/
```
### Parking:
```bash
data/parking/images/
```

You can select the desired dataset inside the code as described further down.

## Running the pipeline
To run the pipeline, perform the following steps:
1. Navigate to the base directory of the repo (where this readme is also located) and install the anaconda environment from the provided environment.yml file:

```bash
conda env create -f environment.yml
```

2. Activate the conda environment:
```bash
conda activate VAMR_Project
```

3. Then run the following command:
```bash
python3 main.py
```

Alternatively, when opening in VS Code, select the created environment as your interpreter and press the Run button while having the main.py file open.

This will run the entire pipeline and create the intermediate plots as well as the final metric plot.
The dataset can be selected by changing the dataset integer in line 15 of the main.py file:
```bash
...
# Setup
dataset = 1 # 0: KITTI, 1: Malaga, 2: parking, 3: test <---- select desired dataset here
...
```