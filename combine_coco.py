import os
import os.path as osp
import json
import shutil
from tqdm import tqdm


root_dir = './'
data_dir = '/Users/shiuchung/temp_chinajin/'

jsons_dir = osp.join(root_dir, 'test_json') #需要合并的json所在目录
imgs_dir = osp.join(data_dir, 'images')
dst_anno_dir = osp.join(root_dir, 'coco_annos')
os.makedirs(dst_anno_dir, exist_ok=True)

categories = []         # coco categories字段，list of dict
labels = ['box'] #自定义
supercategory = ''
for i, label in enumerate(labels):
    category = {}
    category['id'] = i + 1
    category['name'] = label
    category['supercategory'] = supercategory
    categories.append(category)

label_dict = {}         # category_id和label对应信息
for i, label in enumerate(labels):
    label_dict[label] = i + 1

image_id = 0
anno_id = 1
images = []             # coco images字段，图片文件信息
annotations = []        # coco annotations字段，标注信息


img_files = [osp.join(imgs_dir, j) for j in os.listdir(imgs_dir) if osp.splitext(j)[-1] == '.png']

img_files.sort()

#assert len(img_files) == len(json_files)
print(f"{len(img_files)} ==  {len(img_files)}")


labels_info = dict()


for js in os.listdir(jsons_dir):
    print(js)
    with open(osp.join(jsons_dir, js) ,'r',encoding='utf-8') as fp:
        json_info = json.load(fp)
    for imgt in tqdm(json_info['images']):
        try:
            image = dict()
            image['coco_url'] = ''
            image['date_captured'] = ''
            image['flickr_url'] = ''
            image['license'] = 0

            image['id'] = image_id
            image_id += 1

            image['file_name'] = imgt['file_name']
            # print(osp.join(data_dir, 'images', image['file_name']))
            assert osp.exists(osp.join(imgs_dir, image['file_name']))
            image['height'] = imgt['height']
            image['width'] = imgt['width']
            
            images.append(image)
            

            for annt in json_info['annotations']: 
                if annt['image_id'] == imgt['id']:
                    
                    annotation = dict()

                    annotation["category_id"] = annt['category_id']
                    annotation["id"] = anno_id
                    anno_id += 1
                    annotation["image_id"] = image_id - 1
                    annotation["iscrowd"] = 0

                    annotation["segmentation"] = annt['segmentation']
                    annotation["area"] = annt['area']
                    annotation["bbox"] = annt['bbox']
                    
                    annotations.append(annotation)
        except:
            print(imgt['id'])



attrDict = dict()
attrDict['categories'] = categories
attrDict['images'] = images
attrDict['annotations'] = annotations

json_string = json.dumps(attrDict, ensure_ascii=False)
with open(osp.join(dst_anno_dir, 'annos.json'), 'w',encoding='utf-8') as f:
    f.write(json_string)
   

print(len(categories), len(images), len(annotations))

for key, value in labels_info. items():
    print(f"{key}: {len(value)}")

