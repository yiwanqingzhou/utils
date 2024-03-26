import os
import os.path as osp
import json
import shutil
from tqdm import tqdm
from datetime import datetime


def combine_coco(jsons_dir, imgs_dir, save_path, labels, description):

    if os.path.exists(save_path) and not os.path.isdir(save_path):
        raise Exception(
            "Save path exists but isn't a directory: {}".format(save_path)
        )
    os.makedirs(save_path, exist_ok=True)

    now = datetime.now()
    info = {
        "contributor": "",
        "date_created": now.strftime("%Y-%m-%d"),
        "description": description + " {}".format(
            now.strftime("%Y-%m-%d %H-%M-%S")
        ),
        "url": "",
        "version": 1,
        "year": now.strftime("%Y"),
    }

    licenses = [
        {
            "name": "Copyright Dorabot",
            "id": 0,
            "url": "https://www.dorabot.com/",
        }
    ]

    categories = []
    supercategory = ''
    for i, label in enumerate(labels):
        category = {}
        category['id'] = i + 1
        category['name'] = label
        category['supercategory'] = supercategory
        categories.append(category)

    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i + 1

    image_id = 0
    anno_id = 1
    images = []
    annotations = []

    json_files = os.listdir(jsons_dir)
    json_files.sort()

    for js in json_files:
        print(js)
        with open(osp.join(jsons_dir, js), 'r', encoding='utf-8') as fp:
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
    attrDict['info'] = info
    attrDict['licenses'] = licenses
    attrDict['categories'] = categories
    attrDict['images'] = images
    attrDict['annotations'] = annotations

    json_string = json.dumps(attrDict, ensure_ascii=False)
    annotation_filename = os.path.join(
        save_path, now.strftime("annotations_%Y%m%d_%H%M%S.json")
    )
    with open(annotation_filename, 'w', encoding='utf-8') as f:
        f.write(json_string)

    print(len(categories), len(images), len(annotations))


def main():
    root_dir = './'
    jsons_dir = osp.join(root_dir, 'result_json')
    imgs_dir = osp.join(root_dir, 'images')
    save_path = root_dir
    labels = ['box']

    description = 'Chinajin'

    combine_coco(jsons_dir, imgs_dir, save_path, labels, description)


if __name__ == "__main__":
    main()
