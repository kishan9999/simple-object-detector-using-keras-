# import the necessary packages
from bs4 import BeautifulSoup
import random
import os, csv
import numpy as np

# Define Directory 
datasets_directory = "./datasets"
annotations_directory="./datasets/annotations/"

format='.jpg'
N = []      
for r, d, f in os.walk(datasets_directory, topdown=False):
   N.append(f)   

# K=range(len(N[0]))
K=np.arange(len(N[0]))
random.shuffle(K)

for i in K:  
    annotation_file=annotations_directory+N[0][i]
    ds = BeautifulSoup(open(annotation_file).read(), "html.parser")
    w = int(ds.find("width").string)
    h = int(ds.find("height").string)
    
    # Iterating each object elements
    for o in ds.find_all("object"):
        class_label = o.find("name").string
        x_min = max(0, int(float(o.find("xmin").string)))
        y_min = max(0, int(float(o.find("ymin").string)))
        x_max = min(w, int(float(o.find("xmax").string)))
        y_max = min(h, int(float(o.find("ymax").string)))
        # controlling errors
        if x_min >= x_max or y_min >= y_max:
            continue
        elif x_max <= x_min or y_max <= y_min:
            continue
        line = [N[1][i], str(x_min), str(y_min), str(x_max), str(y_max), str(class_label)]
        with open("datasets.csv", 'a', newline='') as f:
                csv.writer(f).writerow(line)

