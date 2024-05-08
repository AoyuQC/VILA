try:
    import debugpy

    debugpy.listen(5888)  # 5678 is port
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    debugpy.breakpoint()
    print('break on this line')
except:
    print("non debug mode")

import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# # g5
# root_dir="/home/ubuntu/dataset/aigc-vlm/ltcc"
# split_dir="split_imgs_0505"
# concat_dir="concat_imgs_0505"
# save_path = f"{root_dir}/pickle/"

# a100
root_dir="/home/ec2-user/SageMaker/data/dataset"
split_dir="split_imgs_0505"
concat_dir="concat_imgs_0505"
save_path = f"{root_dir}/pickle/"

# img_description = json.loads(open("/home/ec2-user/SageMaker/data/res_ltcc.json").read())
img_description = json.loads(open(f"{root_dir}/res_ltcc.json").read())
# for item in img_description:
#     item["pic"] = item["pic"].replace("../","/home/ec2-user/SageMaker/data/") #改成绝对路径
for item in img_description:
    item["pic"] = item["pic"].replace("../",f"{root_dir}/") #改成绝对路径

img_description_map = {}
for item in img_description:
    img_description_map[item['pic']] = item['text']
print(img_description[:2])
print(len(img_description_map.keys()))

import glob, os
import random
from collections import defaultdict

# root_dir = "/home/ec2-user/SageMaker/data/LTCC_ReID/train"
root_dir_train = f"{root_dir}/LTCC_ReID/train"

img_files = glob.glob(os.path.join(root_dir_train,'*.png'))
img_labels = [(os.path.basename(_).split(".")[0].split("_")[0],os.path.basename(_).split(".")[0].split("_")[1]) for _ in img_files]
img_labels[:1]
pid_img_map = defaultdict(list)
positive_pairs = []
negative_pairs = []

for i in range(len(img_files)):
    pid, cid = img_labels[i]
    pid_img_map[pid].append((cid, img_files[i]))

used = set()
pids = list(pid_img_map.keys())

for pid in pids:
    for i in range(len(pid_img_map[pid])):
        first_img_cid, first_img_path = pid_img_map[pid][i]
        if first_img_path not in img_description_map:
            continue
        # if first_img_path in used:
        #     continue
        for j in range(len(pid_img_map[pid])):
            second_img_cid, second_img_path = pid_img_map[pid][j]
            if second_img_path not in img_description_map:
                continue
            if random.random()<0.015:
                positive_pairs.append((first_img_path,second_img_path))
                    
        
print(len(positive_pairs))

for i in range(len(pids)):
    for j in range(len(pids)):
        if i == j:
            continue
        for cid1, path1 in pid_img_map[pids[i]]:
            for cid2, path2 in pid_img_map[pids[j]]:
                if random.random()<0.0005:
                    if path1 not in img_description_map or path2 not in img_description_map:
                        continue
                    negative_pairs.append((path1, path2))
print(len(negative_pairs))

import os

def analyze_ltcc_path(path):
    path = os.path.basename(path).split('.')[0]
    pid, cid, cloth, ts = path.split('_')
    return pid, cid, cloth

demo_path = negative_pairs[0][0]
print(demo_path)
analyze_ltcc_path(demo_path)

import glob, os
import json
import random

res = []

for i in range(len(positive_pairs)):
    img1, img2 = positive_pairs[i]
    
    sample = {}
    sample["id"] = str(i)
    sample["image"] = [img1,img2]

    # prepare same distinct images
    image1_new_name = os.path.basename(img1).replace('.png','.jpg')
    inter_1_split_path = os.path.join(f'{root_dir}/{split_dir}',image1_new_name)
    image2_new_name = os.path.basename(img2).replace('.png','.jpg')
    inter_2_split_path = os.path.join(f'{root_dir}/{split_dir}',image2_new_name)
    Image.open(path1).convert('RGB').save(inter_1_split_path)
    Image.open(path2).convert('RGB').save(inter_2_split_path)
    complete_path = {}
    complete_path['image1']=inter_1_split_path
    complete_path['image2']=inter_2_split_path
    sample["image_pair"] = complete_path

    conversation = []
    text1, text2 = img_description_map[img1], img_description_map[img2]
    conversation.append({"from":"human","value":"<image>\n <image>\n Your task is to answer if the two people in the two images are the same person. Note that they may be same person with different clothes. Now describe the person in first image in detail:"})
    conversation.append({"from":"gpt","value":text1})
    conversation.append({"from":"human","value":"Describe the person in second image detail:"})
    conversation.append({"from":"gpt","value":text2})
    conversation.append({"from":"human","value":"Are these two images from the same person? Answer with yes or no."})
    conversation.append({"from":"gpt","value":"Yes"})
    sample["conversations"] = conversation
    res.append(sample)

offset = len(positive_pairs)

for i in range(len(negative_pairs)):
    img1, img2 = negative_pairs[i]
    sample = {}
    sample["id"] = str(i+offset)
    sample["image"] = [img1,img2]

    # prepare same distinct images
    image1_new_name = os.path.basename(img1).replace('.png','.jpg')
    inter_1_split_path = os.path.join(f'{root_dir}/{split_dir}',image1_new_name)
    image2_new_name = os.path.basename(img2).replace('.png','.jpg')
    inter_2_split_path = os.path.join(f'{root_dir}/{split_dir}',image2_new_name)
    Image.open(path1).convert('RGB').save(inter_1_split_path)
    Image.open(path2).convert('RGB').save(inter_2_split_path)
    complete_path = {}
    complete_path['image1']=inter_1_split_path
    complete_path['image2']=inter_2_split_path
    sample["image_pair"] = complete_path

    conversation = []
    text1, text2 = img_description_map[img1], img_description_map[img2]
    conversation.append({"from":"human","value":"<image>\n <image>\n Your task is to answer if the two people in the two images are the same person. Note that they may be same person with different clothes. Now describe the person in first image in detail:"})
    conversation.append({"from":"gpt","value":text1})
    conversation.append({"from":"human","value":"Describe the person in second image detail:"})
    conversation.append({"from":"gpt","value":text2})
    conversation.append({"from":"human","value":"Are these two images from the same person? Answer with yes or no."})
    conversation.append({"from":"gpt","value":"No"})
    sample["conversations"] = conversation
    res.append(sample)

offset += len(negative_pairs)

# # with open("/home/ec2-user/SageMaker/data/training_samles_2img_no_concat.json",'w') as output_file:
# with open("/home/ubuntu/dataset/aigc-vlm/ltcc/training_samles_2img_no_concat.json",'w') as output_file:
#     output_file.write(json.dumps(res))

# 重采样一批，只用来回答yes no

import glob, os
import random
from collections import defaultdict

img_files = glob.glob(os.path.join(root_dir_train,'*.png'))
img_labels = [(os.path.basename(_).split(".")[0].split("_")[0],os.path.basename(_).split(".")[0].split("_")[1]) for _ in img_files]
img_labels[:1]
pid_img_map = defaultdict(list)
positive_pairs = []
negative_pairs = []

for i in range(len(img_files)):
    pid, cid = img_labels[i]
    pid_img_map[pid].append((cid, img_files[i]))

used = set()
pids = list(pid_img_map.keys())

for pid in pids:
    for i in range(len(pid_img_map[pid])):
        first_img_cid, first_img_path = pid_img_map[pid][i]
        if first_img_path not in img_description_map:
            continue
        # if first_img_path in used:
        #     continue
        for j in range(len(pid_img_map[pid])):
            second_img_cid, second_img_path = pid_img_map[pid][j]
            if second_img_path not in img_description_map:
                continue
            # if second_img_path in used:
            #     continue
            if first_img_cid != second_img_cid:
                if random.random()<0.0039:
                    positive_pairs.append((first_img_path,second_img_path))
                    # used.add(first_img_path)
                    # used.add(second_img_path)
        
print(len(positive_pairs))

for i in range(len(pids)):
    for j in range(len(pids)):
        if i == j:
            continue
        for cid1, path1 in pid_img_map[pids[i]]:
            for cid2, path2 in pid_img_map[pids[j]]:
                if random.random()<0.00010:
                    if path1 not in img_description_map or path2 not in img_description_map:
                        continue
                    negative_pairs.append((path1, path2))
print(len(negative_pairs))


positive_path=[]
for path1, path2 in tqdm(positive_pairs):
    inter_path = os.path.basename(path1).replace('.png','')+'----'+os.path.basename(path2).replace('.png','')+'.png'
    # inter_path = os.path.join('/home/ec2-user/SageMaker/data/concat_imgs_0315',inter_path)
    inter_path = os.path.join(f'{root_dir}/{concat_dir}',inter_path)
    image_1 = np.array(Image.open(path1).resize((168,336)))
    image_2 = np.array(Image.open(path2).resize((168,336)))
    new_img = np.concatenate((image_1, image_2), axis = 1)
    new_img = Image.fromarray(new_img)
    new_img.save(inter_path)
    # prepare same distinct images
    image1_new_name = os.path.basename(path1).replace('.png','.jpg')
    inter_1_split_path = os.path.join(f'{root_dir}/{split_dir}',image1_new_name)
    image2_new_name = os.path.basename(path2).replace('.png','.jpg')
    inter_2_split_path = os.path.join(f'{root_dir}/{split_dir}',image2_new_name)
    Image.open(path1).convert('RGB').save(inter_1_split_path)
    Image.open(path2).convert('RGB').save(inter_2_split_path)
    complete_path = {}
    complete_path['inter']=inter_path
    complete_path['image1']=inter_1_split_path
    complete_path['image2']=inter_2_split_path

    positive_path.append(complete_path)
    
negative_path=[]
for path1, path2 in tqdm(negative_pairs):
    inter_path = os.path.basename(path1).replace('.png','')+'----'+os.path.basename(path2).replace('.png','')+'.png'
    # inter_path = os.path.join('/home/ec2-user/SageMaker/data/concat_imgs_0315',inter_path)
    inter_path = os.path.join(f'{root_dir}/{concat_dir}',inter_path)
    image_1 = np.array(Image.open(path1).resize((168,336)))
    image_2 = np.array(Image.open(path2).resize((168,336)))
    new_img = np.concatenate((image_1, image_2), axis = 1)
    new_img = Image.fromarray(new_img)
    new_img.save(inter_path)
    # prepare same distinct images
    image1_new_name = os.path.basename(path1).replace('.png','.jpg')
    inter_1_split_path = os.path.join(f'{root_dir}/{split_dir}',image1_new_name)
    image2_new_name = os.path.basename(path2).replace('.png','.jpg')
    inter_2_split_path = os.path.join(f'{root_dir}/{split_dir}',image2_new_name)
    Image.open(path1).convert('RGB').save(inter_1_split_path)
    Image.open(path2).convert('RGB').save(inter_2_split_path)
    complete_path = {}
    complete_path['inter']=inter_path
    complete_path['image1']=inter_1_split_path
    complete_path['image2']=inter_2_split_path
    negative_path.append(complete_path)

offset = 0

for i in range(len(positive_pairs)):
    img1, img2 = positive_pairs[i]
    sample = {}
    sample["id"] = str(i+offset)
    sample["image"] = positive_path[i]['inter']
    sample["image_pair"] = [positive_path[i]['image1'],positive_path[i]['image2']]
    conversation = []
    text1, text2 = img_description_map[img1], img_description_map[img2]
    conversation.append({"from":"human","value":"<image>\n <image>\n These are two images of two people. Are the two people the same person? Answer with yes or no."})
    conversation.append({"from":"gpt","value":"Yes"})
    sample["conversations"] = conversation
    res.append(sample)

offset += len(positive_pairs)

for i in range(len(negative_pairs)):
    img1, img2 = negative_pairs[i]
    sample = {}
    sample["id"] = str(i+offset)
    sample["image"] = negative_path[i]['inter']
    sample["image_pair"] = [negative_path[i]['image1'],negative_path[i]['image2']]
    conversation = []
    text1, text2 = img_description_map[img1], img_description_map[img2]
    conversation.append({"from":"human","value":"<image>\n <image>\n These are two images of two people. Are the two people the same person? Answer with yes or no."})
    conversation.append({"from":"gpt","value":"No"})
    sample["conversations"] = conversation
    res.append(sample)


random.shuffle(res)
print(len(res),res[:5])
offset += i
# # with open("/home/ec2-user/SageMaker/data/training_samles_2img.json",'w') as output_file:
# with open('/home/ubuntu/dataset/aigc-vlm/ltcc/training_samles_2mg.json','w') as output_file:
#     output_file.write(json.dumps(res))

import pickle
filtered_samples = []
cnt = 0
for sample in res:
    filtered_samples.append(
        dict(
            question=sample["conversations"][0]["value"],
            answer=sample["conversations"][1]["value"],
            id="text_flan_%08d" % cnt,
            image=sample['image_pair'],
        )
    )
    cnt += 1


with open(os.path.join(save_path, "text_ltcc_50k_v1.pkl"), "wb") as f:
    pickle.dump(filtered_samples, f)

# len(res)

# import glob, os
# import json

# root_dir = "/home/ec2-user/SageMaker/data/res_ufine_default"

# img_files = glob.glob(os.path.join(root_dir,'*.png'))
# print(len(img_files))
# # res = []

# for i in range(len(img_files)):
#     sample = {}
#     sample["id"] = str(offset+i)
#     sample["image"] = img_files[i]
#     conversation = []
#     with open(img_files[i].replace(".png",".txt")) as txt_file:
#         content = txt_file.readlines()
#         conversation.append({"from":"human","value":"<image>\nDescribe the left person in the image in detail:"})
#         conversation.append({"from":"gpt","value":"".join(content[0].strip().split(":")[1:])})
#         conversation.append({"from":"human","value":"Describe the right person in the image in detail:"})
#         conversation.append({"from":"gpt","value":"".join(content[1].strip().split(":")[1:])})
#         conversation.append({"from":"human","value":"Are the two people the same person?"})
#         conversation.append({"from":"gpt","value":"".join(content[2].strip())})
#     sample["conversations"] = conversation
#     res.append(sample)
    
# offset+=i
# # with open(os.path.join("training_samles.json"),'a') as output_file:
# #     output_file.write(json.dumps(res))

# import glob, os
# import json

# root_dir = "/home/ec2-user/SageMaker/data/res_ufine_positive"

# img_files = glob.glob(os.path.join(root_dir,'*.png'))
# print(len(img_files))
# # res = []

# for i in range(len(img_files)):
#     sample = {}
#     sample["id"] = str(offset+i)
#     sample["image"] = img_files[i]
#     conversation = []
#     with open(img_files[i].replace(".png",".txt")) as txt_file:
#         content = txt_file.readlines()
#         conversation.append({"from":"human","value":"<image>\nDescribe the left person in the image in detail:"})
#         conversation.append({"from":"gpt","value":"".join(content[0].strip().split(":")[1:])})
#         conversation.append({"from":"human","value":"Describe the right person in the image in detail:"})
#         conversation.append({"from":"gpt","value":"".join(content[1].strip().split(":")[1:])})
#         conversation.append({"from":"human","value":"Are the two people the same person?"})
#         conversation.append({"from":"gpt","value":"".join(content[2].strip())})
#     sample["conversations"] = conversation
#     res.append(sample)

# offset+=i
# with open(os.path.join("training_samles.json"),'a') as output_file:
#     output_file.write(json.dumps(res))

