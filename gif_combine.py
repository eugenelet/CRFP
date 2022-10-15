import os
from PIL import Image,ImageSequence
import numpy as np
import cv2
from imageio import imread, imsave, get_writer

im_1 = Image.open('DCN_3and1.gif')
im_2 = Image.open('DCN_4.gif')

def iter_frames(im):
    try:
        i= 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0: 
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass

gif_1 = []
gif_2 = []
n_frames = 0
for i,frame in enumerate(ImageSequence.Iterator(im_1),1):
    frame = frame.convert('RGB')
    # frame.save(os.path.join('test.png'))
    # gif_1.append(np.array(Image.open('test.png')))
    gif_1.append(np.array(frame))
    n_frames += 1
    # cv2.imshow('image', gif_1[-1][:,:,::-1])
    # cv2.waitKey(10)

for i,frame in enumerate(ImageSequence.Iterator(im_2),1):
    frame = frame.convert('RGB')
    frame.save(os.path.join('test.png'))
    # gif_2.append(np.array(Image.open('test.png')))
    gif_2.append(np.array(frame))
    # cv2.imshow('image', gif_2[-1][:,:,::-1])
    # cv2.waitKey(10)

with get_writer('test_1.gif', mode="I", fps=7) as writer:
    for n in range(n_frames - 10):
        writer.append_data(np.concatenate((gif_2[n][260:620,270:590,:],gif_1[n][260:620,270:590,:]), axis=1))
        cv2.imshow('image', np.concatenate((gif_2[n][260:620,270:590,:],gif_1[n][260:620,270:590,:]), axis=1))
        cv2.waitKey(100)

# with get_writer('test_1.gif', mode="I", fps=7) as writer:
#     for n in range(n_frames):
#         writer.append_data(gif_1[n][260:620,220:640,:])
