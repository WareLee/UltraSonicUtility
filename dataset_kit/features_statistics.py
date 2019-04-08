import cv2
import os

path = r'F:\workspace\ultrasonic\hnuMedical2\ImageWare\merged_2500'
a, b, c = (0, 0, 0)
x, y, z = (0, 0, 0)
count = 0
for cls in os.listdir(path):
    sub_cls_p = os.path.join(path,cls)
    if not os.path.isdir(sub_cls_p):
        continue
    for img in os.listdir(sub_cls_p):
        im = cv2.imread(os.path.join(sub_cls_p, img))
        mean, stddv = cv2.meanStdDev(im)
        a += float(mean[0])
        b += float(mean[1])
        c += float(mean[2])

        x += float(stddv[0])
        y += float(stddv[1])
        z += float(stddv[2])

        count += 1

print('count: ',count)
print(a, b, c, sep=' ')
print([ch/count for ch in (a,b,c)])
print(x, y, z, sep='  ')
print([dv/count for dv in (x,y,z)])

# count:  16447
# 651796.3872563976 646362.2494590244 615526.7596333406
# [39.63010805960951, 39.299705080502484, 37.42486530268989]
# 764554.1730665533  742436.4510013349  715239.0687632312
# [46.48593500739061, 45.14114738258253, 43.48750950101728]
