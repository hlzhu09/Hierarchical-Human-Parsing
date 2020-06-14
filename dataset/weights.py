import os

import numpy as np
from PIL import Image

# data_root = '../data/CCF/Segmentations/'
# fid = open('./CCF/train_id.txt', 'r')
# num_cls = 18


# data_root = '../data/Person/SegmentationPart/'
# fid = open('./Pascal/train_id.txt', 'r')
# num_cls = 7
data_root = '../data/LIP/train_set/segmentations/'
fid = open('./LIP/train_id.txt', 'r')
num_cls = 20

cls_pix_num = np.zeros(num_cls)
cls_hbody_num = np.zeros(3)
cls_fbody_num = np.zeros(2)

map_idx = [0, 9, 19, 29, 50, 39, 60, 62]

for line in fid.readlines():
    img_path = os.path.join(data_root, line.strip() + '.png')
    # img_data = np.asarray(Image.open(img_path).convert('L'))
    img_data = np.array(Image.open(img_path))
    # for i in range(len(map_idx)):
    #     img_data[img_data == map_idx[i]] = i
    # img_size = img_data.size
    for i in range(num_cls):
        cls_pix_num[i] += (img_data == i).astype(int).sum(axis=None)

# # half body
# cls_hbody_num[0] = cls_pix_num[0]
# for i in range(1, 5):
#     cls_hbody_num[1] += cls_pix_num[i]
# for i in range(5, 8):
#     cls_hbody_num[2] += cls_pix_num[i]
#
# # full body
# cls_fbody_num[0] = cls_pix_num[0]
# for i in range(1, 8):
#     cls_fbody_num[1] += cls_pix_num[i]

weight = np.log(cls_pix_num)
weight_norm = np.zeros(num_cls)
for i in range(num_cls):
    weight_norm[i] = 16 / weight[i]
print(weight_norm)


# [0.8373, 0.918, 0.866, 1.0345, 1.0166,
#  0.9969, 0.9754, 1.0489, 0.8786, 1.0023,
#  0.9539, 0.9843, 1.1116, 0.9037, 1.0865,
#  1.0955, 1.0865, 1.1529, 1.0507]

# 0.93237515, 1.01116892, 1.11201307

# 0.98417377, 1.05657165

# ATR training
# [0.85978634, 1.19630769, 1.02639146, 1.30664970, 0.97220603, 1.04885815,
#  1.01745278, 1.01481690, 1.27155077, 1.12947663, 1.13016390, 1.06514227,
#  1.08384483, 1.08506841, 1.09560942, 1.09565198, 1.07504567, 1.20411509]

#CCF
# [0.82073458, 1.23651165, 1.0366326,  0.97076566, 1.2802332,  0.98860602,
#  1.29035071, 1.03882453, 0.96725283, 1.05142434, 1.0075884,  0.98630539,
#  1.06208869, 1.0160915,  1.1613597,  1.17624919, 1.1701143,  1.24720215]

#PPSS
# [0.89680465, 1.14352656, 1.20982646, 0.99269248,
#  1.17911144, 1.00641032, 1.47017195, 1.16447113]

#Pascal
# [0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914]

#Lip
# [0.7602572,  0.94236198, 0.85644457, 1.04346266, 1.10627293, 0.80980162,
#  0.95168713, 0.8403769,  1.05798412, 0.85746254, 1.01274366, 1.05854692,
#  1.03430773, 0.84867818, 0.88027721, 0.87580925, 0.98747462, 0.9876475,
#  1.00016535, 1.00108882]
