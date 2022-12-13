
import numpy as np
import re
import os
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage
import cv2

class CreateImgsDataset():
  def __init__(self):
    self.image_directories = []

  def traverse_img(self, folder_path): # путь до нужной директории
      files, dirs = self.get_files_and_dirs(folder_path)
      has_json = self.check_json_file_existance(files)
      if has_json: # директория с изображениями лежит в папке с json файлом
        paths = list(map(lambda dir: os.path.join(folder_path, dir), dirs))
        self.image_directories.extend(paths)
      else:
        for dir in dirs:
            self.traverse_img(os.path.join(folder_path, dir))
      return self.image_directories

  def get_imgs_dataset(self, root):
    directories = self.traverse_img(root)
    directories.sort()

    for dir in directories:
      try:
        img_iter = self.load_dicom(dir)
        imgs = np.append(imgs, img_iter, axis=0)
      except:
        imgs = self.load_dicom(dir)

    return imgs

  # вспомогательные функции
  def get_files_and_dirs(self, path):
    files = []
    dirs = []
    for entry in os.scandir(path):
      if entry.name == '.ipynb_checkpoints' or entry.name == '.DS_Store':
        continue
      if entry.is_dir():
          dirs.append(entry.name)
      else:
        files.append(entry.name)

    return files, dirs

  def is_name_matches_json(self, filename):
    regexp = "([a-zA-Z0-9\s_\\.\-\(\):])+(.json)$"
    return len(re.findall(regexp, filename)) > 0;

  def check_json_file_existance( self,filenames):
    return len(list(filter(lambda f: self.is_name_matches_json(f), filenames))) > 0

  def load_dicom(self, directory):
      reader = sitk.ImageSeriesReader()
      dicom_names = reader.GetGDCMSeriesFileNames(directory)
      reader.SetFileNames(dicom_names)
      image_itk = reader.Execute()

      image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
      return image_zyx

class CreateMasksDataset():
  def __init__(self):
    self.mask_directories = [] 

  def traverse_mask(self, folder_path): # путь до файолов с масками
    files, dirs = self.get_files_and_dirs(folder_path)
    if not dirs:
      paths = list(map(lambda fil: os.path.join(folder_path, fil), files))
      self.mask_directories.extend(paths)
    else:
      for dir in dirs:
        self.traverse_mask(os.path.join(folder_path, dir))
    return self.mask_directories

  def get_masks_dataset(self, root):
    mask_paths = self.traverse_mask(root)
    mask_paths.sort()

    for mask_path in mask_paths:
      try:
        mask_itr = nib.load(mask_path)
        mask_itr = mask_itr.get_fdata().transpose(2, 0, 1)
        mask_itr = scipy.ndimage.rotate(mask_itr, 90, (1, 2))
        masks = np.append(masks, mask_itr, axis=0)
      except:
        masks = nib.load(mask_path)
        masks = masks.get_fdata().transpose(2, 0, 1)
        masks = scipy.ndimage.rotate(masks, 90, (1, 2))

    return masks

  def get_files_and_dirs(self, path):
    files = []
    dirs = []
    for entry in os.scandir(path):
      if entry.name == '.ipynb_checkpoints' or entry.name == '.DS_Store':
        continue
      if entry.is_dir():
          dirs.append(entry.name)
      else:
        files.append(entry.name)

    return files, dirs

class Preprocessing(): #resize + cut + normalize
  def __init__(self):
    self.W_size, self.H_size = 128, 128
    self.len = 1087
    self.l_cut, self.r_cut = 80, 432
    self.data_imgs, self.data_masks = np.empty((2, self.len, self.W_size, self.H_size), dtype=np.float32)

  def preprocess_img(self, imgs_datatset):
    for i, img in enumerate(imgs_datatset):
      img = img.astype('float32')
      img = img[self.l_cut:self.r_cut, self.l_cut:self.r_cut]
      img = cv2.resize(img, dsize=(self.W_size, self.H_size))
      img = (img - np.min(img)) / (np.max(img) - np.min(img)) #normalize

      self.data_imgs[i] = np.expand_dims(img, axis = 0)
    return self.data_imgs

  def preprocess_mask(self, masks_dataset):
    for i, mask in enumerate(masks_dataset):
      mask = mask.astype('float32')
      mask = mask[self.l_cut:self.r_cut, self.l_cut:self.r_cut]
      mask = cv2.resize(mask, dsize=(self.W_size, self.H_size))
      mask[mask < 0] = 0

      self.data_masks[i] = np.expand_dims(mask, axis = 0)
    return self.data_masks