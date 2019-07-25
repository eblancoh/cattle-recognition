from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from glob import glob
import argparse


def compare_images(imageA, imageB, height, width):
    imageA = cv2.imread(imageA)
    imageB = cv2.imread(imageB)
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    imageA = cv2.resize(imageA, (height, width))
    imageB = cv2.resize(imageB, (height, width))

    s = measure.compare_ssim(imageA, imageB)
    return s

def list_images(dir):
    ext = ['.png', '.jpg', '.jpeg']
    glob_pattern = os.path.join(dir, '*')
    files = sorted(glob(glob_pattern), key=os.path.getctime)
    files = [x for x in files if x.endswith(tuple(ext))]
    return files

def clean_dataset(files, threshold):
    unmasked = list()
    for k in range(len(files)):
        ref = './' + files[k].replace('\\', '/')
        targets = [x for i,x in enumerate(files) if i!=k]
        for img in targets:
            img = './' + img.replace('\\', '/')
            s = compare_images(ref, img, height=224, width=224)
            if s > threshold:
                print('Reference Image: ', ref, ' Image to delete:', img, 'with SSIM: ', round(s, 5))
                unmasked.append([ref, img])
    
    unmasked = unmasked[:int(len(unmasked)/2)]
    aux = list()
    for item in unmasked:
        aux.append(item[1])
    aux = np.unique(aux)
    for g in aux:
        os.remove(g)

def main():
    parser = argparse.ArgumentParser(
        description="Computing the Structural Similarity Image Metric (SSIM) of dataset to remove very similar samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('--dir', '-dir', type=str, help='Directory of the images to analyze')
    parser.add_argument('--threshold', '-thres', type=float, default=None, help='SSIM threshold (between 0 and 1)')
    args = parser.parse_args()

    if not args.dir:
        parser.print_help()
        raise ValueError('Especifique un directorio para hacer el análisis de las imágenes')
    if not args.threshold:
        parser.print_help()
        raise ValueError('Especifique un límite numérico para descartar imágenes')
    
    unmasked = clean_dataset(files=list_images(args.dir), threshold=args.threshold)
    
    print('¡Hecho! Filtrado de imágenes más similares con un SSIM score superior a ', args.threshold, 'terminado')

if __name__ == "__main__":
    main()