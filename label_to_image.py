import json
import numpy as np
import os
import glob
import cv2

# 读取labelme json文件
label_dir = '/mnt2/shaobo/evimo/sunny_record/frames/'
save_dir = '/mnt2/shaobo/evimo/sunny_record/label_images'
def label_to_image(data_name):
    json_files = glob.glob(os.path.join(label_dir, data_name, "*.json"))
    for json_file in json_files:
        file_index = os.path.basename(json_file)
        file_index = os.path.splitext(file_index)[0]
        with open(json_file, 'r') as f:
            label_data = json.load(f)

        # 获取标记为1的区域
        shapes = label_data['shapes']
        mask = []
        for shape in shapes:
            if shape['label'] == '1':
                points = shape['points']
                mask.append(points)

        # 创建黑白图像，将标记为1的区域设置为白色
        height, width = 262, 320
        img = np.zeros((height, width), np.uint8)
        for pts in mask:
            pts = np.array(pts, np.int32)
            cv2.fillPoly(img, [pts], 255)

        # 将图像转换为灰度图
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img[3:259]

        # 显示结果
        data_save_path = os.path.join(save_dir, data_name)
        if not os.path.exists(data_save_path):
            os.mkdir(data_save_path)
        cv2.imwrite(os.path.join(save_dir, data_name, file_index + '.png'), gray)

if __name__ == '__main__':
    sample = ('book_1', 'book_2', 'box', 'hdr_walking',
              'w_c_2', 'w_c_3', 'walking', 'walk and catch')
    json_files = glob.glob(os.path.join(label_dir, "*", "*.json"))
    print(len(json_files))

    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for data_name in sample:
        label_to_image(data_name)
    '''

    