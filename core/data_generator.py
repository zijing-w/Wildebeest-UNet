#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import gdal
import rasterio
from rasterio import windows
from rasterio.windows import Window


# In[2]:


class DataGenerator(object):
  def __init__(self, image_path, label_path, test_image_path, test_label_path, input_image_channel=[1,2,3], patchsize=336):
    self.image_path = image_path
    self.label_path = label_path
    self.test_image_path = test_image_path
    self.test_label_path = test_label_path
    self.input_image_channel = input_image_channel
    self.NBANDS = len(self.input_image_channel)
    self.patchsize = patchsize
    self.image_list = self.load_image(self.image_path)
    self.label_list = self.load_label(self.label_path)
    self.train_meta_list = self.load_meta(self.image_path, patchsize)
    self.test_image_list = self.load_image(self.test_image_path)
    self.test_meta_list = self.load_meta(self.test_image_path, patchsize)
    self.test_label_list = self.load_label(self.test_label_path)


  def load_image(self, image_path):
    image_list={}
    for f in sorted(os.listdir(image_path)):
      fdir = os.path.join(image_path, f)
      _, ext = os.path.splitext(f)
      if ext.lower() == ".tif":
          imgNo = _
          #print(imgtype)
          image_data=gdal.Open(fdir)
          bands = [image_data.GetRasterBand(i+1).ReadAsArray() for i in range(self.NBANDS)]
          image_list[imgNo] = np.stack(bands, axis=2)
    return image_list

  def load_label(self, label_path):
    label_list={}
    for f in sorted(os.listdir(label_path)):
      fdir = os.path.join(label_path, f)
      _, ext = os.path.splitext(f)
      if ext.lower() == ".tif":
#         labelNo = f[-7:-4]
        labelNo = _
        #print(imgtype)
        label_data=gdal.Open(fdir)
        bands = [label_data.GetRasterBand(i+1).ReadAsArray() for i in range(label_data.RasterCount)]
        label_list[labelNo] = np.stack(bands, axis=2)
    return label_list

  def load_meta(self, test_image_path, patchsize):
    meta_list=[]
    for f in sorted(os.listdir(test_image_path)):
      fdir = os.path.join(test_image_path, f)
      _, ext = os.path.splitext(f)
      if ext.lower() == ".tif":
        src=rasterio.open(fdir)
        
        nrows, ncols = src.shape
        nbands = 1
        
        for i in range(int(nrows/patchsize)):
            for j in range(int(ncols/patchsize)):
                src=rasterio.open(fdir)
                # Create a Window and calculate the transform from the source dataset    
                window = Window(i*patchsize, j*patchsize, patchsize, patchsize)
                transform = src.window_transform(window)
                # Create a new cropped raster to write to
                profile = src.profile
                profile.update({
                  'height': patchsize,
                  'width': patchsize,
                  'transform': transform})
                meta_list = np.append(meta_list, src)   


#         src=rasterio.open(fdir)
#         # Create a Window and calculate the transform from the source dataset    
#         window = Window(0, 0, patchsize, patchsize)
#         transform = src.window_transform(window)
#         # Create a new cropped raster to write to
#         profile = src.profile
#         profile.update({
#           'height': patchsize,
#           'width': patchsize,
#           'transform': transform})
#         meta_list = np.append(meta_list, src)       
    
#         src=rasterio.open(fdir)
#         # Create a Window and calculate the transform from the source dataset    
#         window = Window(0, 290, patchsize, patchsize)
#         transform = src.window_transform(window)
#         # Create a new cropped raster to write to
#         profile = src.profile
#         profile.update({
#           'height': patchsize,
#           'width': patchsize,
#           'transform': transform})
#         meta_list = np.append(meta_list, src)  
        
        
#         src=rasterio.open(fdir)
#         # Create a Window and calculate the transform from the source dataset    
#         window = Window(290, 0, patchsize, patchsize)
#         transform = src.window_transform(window)
#         # Create a new cropped raster to write to
#         profile = src.profile
#         profile.update({
#           'height': patchsize,
#           'width': patchsize,
#           'transform': transform})
#         meta_list = np.append(meta_list, src)    
        
#         src=rasterio.open(fdir)
#         # Create a Window and calculate the transform from the source dataset    
#         window = Window(290, 290, patchsize, patchsize)
#         transform = src.window_transform(window)
#         # Create a new cropped raster to write to
#         profile = src.profile
#         profile.update({
#           'height': patchsize,
#           'width': patchsize,
#           'transform': transform})
#         meta_list = np.append(meta_list, src)  
    
    return meta_list

  def gridwise_sample(self, imgarray, patchsize):
    """Extract sample patches of size patchsize x patchsize from an image (imgarray) in a gridwise manner.
    """
    nrows, ncols, nbands = imgarray.shape
    patchsamples = np.zeros(shape=(0, patchsize, patchsize, nbands),
                            dtype=imgarray.dtype)

    
    
    for i in range(int(nrows/patchsize)):
      for j in range(int(ncols/patchsize)):
        tocat = imgarray[i*patchsize:(i+1)*patchsize,
                        j*patchsize:(j+1)*patchsize, :]
        tocat = np.expand_dims(tocat, axis=0)
        patchsamples = np.concatenate((patchsamples, tocat),
                                      axis=0)


#     tocat = imgarray[0:336,
#                     0:336, :]
#     tocat = np.expand_dims(tocat, axis=0)
#     patchsamples = np.concatenate((patchsamples, tocat),
#                                   axis=0)
#     tocat = imgarray[0:336,
#                     290:626, :]
#     tocat = np.expand_dims(tocat, axis=0)
#     patchsamples = np.concatenate((patchsamples, tocat),
#                                   axis=0)
#     tocat = imgarray[290:626,
#                     0:336, :]
#     tocat = np.expand_dims(tocat, axis=0)
#     patchsamples = np.concatenate((patchsamples, tocat),
#                                   axis=0)
#     tocat = imgarray[290:626,
#                     290:626, :]
#     tocat = np.expand_dims(tocat, axis=0)
#     patchsamples = np.concatenate((patchsamples, tocat),
#                                   axis=0)
    


    return patchsamples

  #Convert training dataset into patches
  def generate_patches(self):
    self.Xtrain = np.zeros(shape=(0, self.patchsize, self.patchsize, self.NBANDS), dtype=np.uint8)
    self.Ytrain = np.zeros(shape=(0, self.patchsize, self.patchsize, 1), dtype=np.uint8)

    for area in self.image_list.keys():
      X_toadd = self.gridwise_sample(self.image_list[area], self.patchsize)
      Y_toadd = self.gridwise_sample(self.label_list[area], self.patchsize)
      self.Xtrain = np.concatenate((self.Xtrain, X_toadd), axis=0)
      self.Ytrain = np.concatenate((self.Ytrain, Y_toadd), axis=0)

    print("There are %i number of training patches" % (self.Xtrain.shape[0]))

    #Convert testing dataset into patches
    self.Xtest = np.zeros(shape=(0, self.patchsize, self.patchsize, self.NBANDS), dtype=np.uint8)
    self.Ytest = np.zeros(shape=(0, self.patchsize, self.patchsize, 1), dtype=np.uint8)
    for area in self.test_image_list.keys():
      X_toadd = self.gridwise_sample(self.test_image_list[area], self.patchsize)
      Y_toadd = self.gridwise_sample(self.test_label_list[area], self.patchsize)
      self.Xtest = np.concatenate((self.Xtest, X_toadd), axis=0)
      self.Ytest = np.concatenate((self.Ytest, Y_toadd), axis=0)

    print("There are %i number of testing patches" % (self.Xtest.shape[0]))

    data_patch = {
        'Xtrain': self.Xtrain,
        'Ytrain': self.Ytrain,
        'Xtest': self.Xtest,
        'Ytest': self.Ytest
        }
    return data_patch


# In[ ]:




