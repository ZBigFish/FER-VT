#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import os
import csv
import argparse
import numpy as np
from itertools import islice
from PIL import Image
import cv2
import augment

# List of folders for training, validation and test.
folder_names = {'Training': 'FER2013Train',
                'PublicTest': 'FER2013Valid',
                'PrivateTest': 'FER2013Test'}
albumentations = augment.Albumentations()

def str_to_image(image_blob):
    ''' Convert a string blob to an image object. '''
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    image_data = np.expand_dims(image_data,2)
    image_data = np.concatenate((image_data, image_data, image_data), axis=-1)
    return image_data


def main(base_folder, fer_path, ferplus_path,augments):
    '''
    Generate PNG image files from the combined fer2013.csv and fer2013new.csv file. The generated files
    are stored in their corresponding folder for the trainer to use.
    
    Args:
        base_folder(str): The base folder that contains  'FER2013Train', 'FER2013Valid' and 'FER2013Test'
                          subfolder.
        fer_path(str): The full path of fer2013.csv file.
        ferplus_path(str): The full path of fer2013new.csv file.
    '''

    print("Start generating ferplus images.")

    for key, value in folder_names.items():
        folder_path = os.path.join(base_folder, value)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    ferplus_entries = []
    with open(ferplus_path, 'r') as csvfile:
        ferplus_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(ferplus_rows, 1, None):
            ferplus_entries.append(row)

    index = 0
    print("Done...")
    with open(fer_path, 'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(fer_rows, 1, None):
            ferplus_row = ferplus_entries[index]
            # file_name = str(index).strip()
            file_name = ferplus_row[12].strip() + '_' + ferplus_row[1].strip()
            if len(ferplus_row[1].strip()) > 0 and int(ferplus_row[12].strip()) != 9 and int(
                    ferplus_row[12].strip()) != 10:
                image = str_to_image(row[1])
                image_path = os.path.join(base_folder, folder_names[row[2]], file_name)
                cv2.imwrite(image_path,image)
                augimg_path = os.path.join(base_folder, folder_names[row[2]])
                if augments==1 and row[2]=='Training':
                    aug_img = albumentations(image)
                    cv2.imwrite(augimg_path+'/'+ferplus_row[12].strip() + '_augment-ud_' + ferplus_row[1].strip(), np.flipud(aug_img))
                    cv2.imwrite(augimg_path+'/'+ferplus_row[12].strip() + '_augment-lr_' + ferplus_row[1].strip(), np.fliplr(aug_img))
                    augment.augment_hsv(aug_img)
                    cv2.imwrite(augimg_path+'/'+ferplus_row[12].strip() + '_augment-hsv-per_' + ferplus_row[1].strip(), augment.random_perspective(aug_img))
                    cv2.imwrite(augimg_path+'/'+ferplus_row[12].strip() + '_augment-hsv-ud_' + ferplus_row[1].strip(), np.flipud(aug_img))
                    cv2.imwrite(augimg_path+'/'+ferplus_row[12].strip() + '_augment-hsv_' + ferplus_row[1].strip(), aug_img)
                    cv2.imwrite(augimg_path+'/'+ferplus_row[12].strip() + '_augment-hsv-lr_' + ferplus_row[1].strip(), np.fliplr(aug_img))
                if augments == 2 and row[2] == 'Training':
                    if int(ferplus_row[12].strip())==6 or int(ferplus_row[12].strip())==8:
                        aug_img = albumentations(image)
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_ud_' + ferplus_row[1].strip(),
                            np.flipud(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_lr_' + ferplus_row[1].strip(),
                            np.fliplr(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_per_' + ferplus_row[1].strip(),
                            augment.random_perspective(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_udlr_' + ferplus_row[1].strip(),
                            np.fliplr(np.flipud(aug_img)))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_udper_' + ferplus_row[1].strip(),
                            augment.random_perspective(np.flipud(aug_img)))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_lrper_' + ferplus_row[1].strip(),
                            augment.random_perspective(np.fliplr(aug_img)))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_udlrper_' + ferplus_row[1].strip(),
                            augment.random_perspective(np.fliplr(np.flipud(aug_img))))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_perud_' + ferplus_row[1].strip(),
                            np.flipud(augment.random_perspective(aug_img)))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_perlr_' + ferplus_row[1].strip(),
                            np.fliplr(augment.random_perspective(aug_img)))
                        augment.augment_hsv(aug_img)
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_hsv_' + ferplus_row[1].strip(),
                            aug_img)
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_udhsv_' + ferplus_row[1].strip(),
                            np.flipud(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_lrhsv_' + ferplus_row[1].strip(),
                            np.fliplr(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_perhsv_' + ferplus_row[1].strip(),
                            augment.random_perspective(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_udlrperhsv_' + ferplus_row[1].strip(),
                            augment.random_perspective(np.fliplr(np.flipud(aug_img))))
                    elif int(ferplus_row[12].strip())==7:
                        aug_img = albumentations(image)
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_ud_' + ferplus_row[1].strip(),
                            np.flipud(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_lr_' + ferplus_row[1].strip(),
                            np.fliplr(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_per_' + ferplus_row[1].strip(),
                            augment.random_perspective(aug_img))
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_udlr_' + ferplus_row[1].strip(),
                            np.fliplr(np.flipud(aug_img)))
                        augment.augment_hsv(aug_img)
                        cv2.imwrite(
                            augimg_path + '/' + ferplus_row[12].strip() + '_hsv_' + ferplus_row[1].strip(),
                            aug_img)

            index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--base_folder",
                        type=str,
                        help="Base folder containing the training, validation and testing folder.",
                        required=True)
    parser.add_argument("-fer",
                        "--fer_path",
                        type=str,
                        help="Path to the original fer2013.csv file.",
                        required=True)

    parser.add_argument("-ferplus",
                        "--ferplus_path",
                        type=str,
                        help="Path to the new fer2013new.csv file.",
                        required=True)

    parser.add_argument("-augment",
                        "--augments",
                        type=int,
                        help="use augment to dataset.1=normal augment，2=with data num enhance augment",
                        required=True)

    args = parser.parse_args()
    main(args.base_folder, args.fer_path, args.ferplus_path, args.augments)
