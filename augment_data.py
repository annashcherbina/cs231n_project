#data augmentation for Tiny ImageNet 
import numpy as np
import skimage 
from skimage.transform import resize 
from skimage import data,exposure,img_as_float
import random 
from PIL import ImageEnhance,Image 

def flip_horizontally(X):
  return X[:,:,:,::-1] 


def flip_vertically(X): 
    return X[:,:,::-1,:] 


def random_crops(X, crop_shape):
  N, C, H, W = X.shape
  HH, WW = crop_shape
  assert HH < H and WW < W

  out = np.zeros((N, C, H, W), dtype=X.dtype)

  np.random.randint((H-HH), size=N) 
  y_start = np.random.randint((H-HH), size=N)
  x_start = np.random.randint((W-WW), size=N)

  for i in xrange(N):
    cropped = X[i, :, y_start[i]:y_start[i]+HH, x_start[i]:x_start[i]+WW]
    out[i]=resize(cropped,(C,H,W),preserve_range=True)
  return out


def random_contrast(X, scale=(0.8, 1.2)):
  """
  Randomly adjust the contrast of images. For each input image, choose a
  number uniformly at random from the range given by the scale parameter,
  and multiply each pixel of the image by that number.
  Inputs:
  - X: (N, C, H, W) array of image data
  - scale: Tuple (low, high). For each image we sample a scalar in the
    range (low, high) and multiply the image by that scaler.
  Output:
  - Rescaled array out of shape (N, C, H, W) where out[i] is a contrast
    adjusted version of X[i].
  """
  low, high = scale
  N = X.shape[0]
  out = np.zeros_like(X)
  l = (scale[1]-scale[0])*np.random.random_sample(N)+scale[0]
  print str(l.shape) 
  print str(l) 
  out = X * l[:,None,None,None]
  return out



def random_tint(X, scale=(-5, 5)):
  low, high = scale
  N, C = X.shape[:2]
  out = np.zeros_like(X)
  l = (scale[1]-scale[0])*np.random.random_sample((N,C))+scale[0]
  out = X+l[:,:,None,None]
  return out


def fixed_crops(X, crop_shape, crop_type):
  """
  Take center or corner crops of images.
  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - crop_shape: Tuple of integers (HH, WW) giving the size to which each
    image will be cropped.
  - crop_type: One of the following strings, giving the type of crop to
    compute:
    'center': Center crop
    'ul': Upper left corner
    'ur': Upper right corner
    'bl': Bottom left corner
    'br': Bottom right corner
  Returns:
  Array of cropped data of shape (N, C, HH, WW) 
  """
  N, C, H, W = X.shape
  HH, WW = crop_shape

  x0 = (W - WW) / 2
  y0 = (H - HH) / 2
  x1 = x0 + WW
  y1 = y0 + HH

  if crop_type == 'center':
    return X[:, :, y0:y1, x0:x1]
  elif crop_type == 'ul':
    return X[:, :, :HH, :WW]
  elif crop_type == 'ur':
    return X[:, :, :HH, -WW:]
  elif crop_type == 'bl':
    return X[:, :, -HH:, :WW]
  elif crop_type == 'br':
    return X[:, :, -HH:, -WW:]
  else:
    raise ValueError('Unrecognized crop type %s' % crop_type)
