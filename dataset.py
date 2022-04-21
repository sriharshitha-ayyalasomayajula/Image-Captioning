import os
from pycocotools.coco import COCO
dataDir = '.'
dataType = 'val2014’                                                                                                                                                                      

instances_annFile = os.path.join(dataDir, 'cocoapi/annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

captions_annFile = os.path.join(dataDir, 'cocoapi/annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

ids = list(coco.anns.keys())



import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
%matplotlib inline


ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']


print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()


annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)