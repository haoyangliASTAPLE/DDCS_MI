import argparse
import os
import shutil
import csv
import random
from tqdm import tqdm

def umdfaces_reformat():
    min_num = 30
    root_folder = './.data/umdfaces/images'
    train_folder = './.data/umdfaces/train'
    test_folder = './.data/umdfaces/test'
    aux_folder = './.data/umdfaces/aux'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(aux_folder):
        os.makedirs(aux_folder)

    sub_folders = [name for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name))]
    # rename the folder
    for sub_folder in sub_folders:
        old_path = os.path.join(root_folder, sub_folder)
        folder_number = int(sub_folder)
        new_subfolder = str(folder_number).zfill(5)
        new_path = os.path.join(root_folder, new_subfolder)
        os.rename(old_path, new_path)

    for sub_folder in os.listdir(root_folder):
        num_imgs = len(os.listdir(os.path.join(root_folder, sub_folder)))
        if num_imgs < min_num:
            print(f'Class {sub_folder} has less than {min_num} samples, removed!')
            shutil.rmtree(os.path.join(root_folder, sub_folder))
    print(len(os.listdir(root_folder)))

    train_subs = os.listdir(root_folder)[:1000]
    aux_subs = os.listdir(root_folder)[1000:2000]

    for train_sub in train_subs:
        path = os.path.join(root_folder, train_sub) # current path

        # move test images
        test_sub = os.path.join(test_folder, train_sub)
        test_imgs = os.listdir(path)[-3:]
        if not os.path.exists(test_sub):
            os.makedirs(test_sub) # make test folder
        for img in test_imgs:
            shutil.move(os.path.join(path, img), test_sub)

        # move train images
        shutil.move(path, train_folder)

    for aux_sub in aux_subs:
        shutil.move(os.path.join(root_folder, aux_sub), aux_folder)

    shutil.rmtree(root_folder)

def stanforddogs_reformat():
    num_test = 15
    root_folder = './.data/stanforddogs/Images'
    train_folder = './.data/stanforddogs/train'
    test_folder = './.data/stanforddogs/test'
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    subfolders = os.listdir(root_folder)
    for i, subfolder in enumerate(subfolders):
        label = str(i).zfill(3)
        # create folder for this label
        train_subfolder = os.path.join(train_folder, label)
        test_subfolder = os.path.join(test_folder, label)
        if not os.path.exists(train_subfolder):
            os.mkdir(train_subfolder)
        if not os.path.exists(test_subfolder):
            os.mkdir(test_subfolder)

        # select test images for this label
        root_subfolder = os.path.join(root_folder, subfolder)
        image_files = os.listdir(root_subfolder)
        image_files.sort()

        # move train samples
        for image_file in image_files[:-num_test]:
            source_path = os.path.join(root_subfolder, image_file)
            target_path = os.path.join(train_subfolder, image_file)
            shutil.move(source_path, target_path)
        # move test samples
        for image_file in image_files[-num_test:]:
            source_path = os.path.join(root_subfolder, image_file)
            target_path = os.path.join(test_subfolder, image_file)
            shutil.move(source_path, target_path)

    shutil.rmtree(root_folder)

def tsinghuadogs_reformat():
    seed = 0
    random.seed(seed)

    root_folder = './.data/tsinghuadogs'
    image_files = []
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for f in os.listdir(subfolder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_name = os.path.join(subfolder, f)
                    image_files.append(image_name)

    num_images = 30000
    selected_images = random.sample(image_files, num_images)

    # delete the images that are not selected
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for f in os.listdir(subfolder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_name = os.path.join(subfolder, f)
                    if image_name not in selected_images:
                        remove_name = os.path.join(root_folder, image_name)
                        os.remove(remove_name)
                        print(f'remove image {remove_name}')

    # delete empty folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            if not os.path.isdir(subfolder_path):
                os.rmdir(subfolder_path)
                print(f'remove folder {subfolder_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='umdfaces')
    args = parser.parse_args()
    if args.dataset == 'umdfaces':
        umdfaces_reformat()
    elif args.dataset == 'stanforddogs':
        stanforddogs_reformat()
    elif args.dataset == 'tsinghuadogs':
        tsinghuadogs_reformat()