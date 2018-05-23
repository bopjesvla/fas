from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
import os
import numpy as np
from numpy import linalg as LA
import time
import math as m

def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

def sample(mu,sigma,_min,_max):
    return max(_min,min(_max,np.random.normal(mu,sigma)))

def fix_rotate(im, degrees, crop=True):
    im = im.rotate( degrees )

    if crop:
        tan_L = m.tan(abs(m.radians(degrees)))
        w,h = im.size
        left   =     (h/2)*tan_L
        top    =     (w/2)*tan_L
        right  = w - (h/2)*tan_L
        bottom = h - (w/2)*tan_L
        im = im.crop((left,top,right,bottom))
    
    return im

def rand_rotate(im, sigma=0.5, minmax=5, crop=True):
    degrees = sample( 0, sigma, -minmax, minmax )
    im = im.rotate( degrees )
    
    if crop:
        tan_L = m.tan(abs(m.radians(degrees)))
        w,h = im.size
        left   =     (h/2)*tan_L
        top    =     (w/2)*tan_L
        right  = w - (h/2)*tan_L
        bottom = h - (w/2)*tan_L
        im = im.crop((left,top,right,bottom))
    
    return im

def resize(im, h):
    old_w, old_h = im.size

    perc_h = ( h / float( old_h ))
    w = int((float( old_w ) * float( perc_h ) ))
    im = im.resize((w,h), Image.ANTIALIAS)
    
    return im

def fixsize(im, size):
    old_w, old_h = im.size

    if old_w > old_h:
        perc_w = ( size / float(old_w) )
        h = int((float( old_h) * float(perc_w)))
        im = im.resize((size,h), Image.ANTIALIAS)
    else:
        perc_h = (size / float(old_h))
        w = int((float( old_w ) * float( perc_h )))
        im = im.resize((w,size), Image.ANTIALIAS)

    old_size = im.size
    new_size = (size,size)
    new_im = Image.new("RGB", new_size)
    new_im.paste(im, (int((new_size[0]-old_size[0])/2),
                      int((new_size[1]-old_size[1])/2)))
    return new_im

def autocrop(im):
    w,h = im.size
    zero   = (0,0,0)
    col_tl = im.getpixel((0,0))
    col_br = im.getpixel((w-1,h-1))

    np_im = np.array(im)
    np_im[(np_im == col_tl ).all(axis=-1)] = zero
    np_im[(np_im == col_br ).all(axis=-1)] = zero

    im2 = Image.fromarray(np_im, mode='RGB')
    im = im.crop( im2.getbbox() )
    return im

def modify(im,rate):
    im = ImageEnhance.Color(im).enhance(      sample( 1,rate,0.5,1.5))
    im = ImageEnhance.Contrast(im).enhance(   sample( 1,rate,0.5,1.5))
    im = ImageEnhance.Brightness(im).enhance( sample( 1,rate,0.5,1.5))
    im = ImageEnhance.Sharpness(im).enhance(  sample( 1,rate,0.5,1.5))
    return im

i = 0
for _,_,im_names in os.walk('./res/train'):
    for im_name in im_names:
        with Image.open('./res/train/'+im_name) as im:
            i += 1
            print(i, end='\r')
            im = modify(im, 0.05)
            im = autocrop(im)
            im = fixsize(im, 512)
            #im = resize(im, 512)
            #im.show()
            #time.sleep(5)

