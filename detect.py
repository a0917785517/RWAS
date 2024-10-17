import torch
import torchvision.transforms as transforms
import argparse
import numpy as np
import os
import sys
import ast
repo_directory = os.getcwd()  # 獲取專案路徑
modules_path = os.path.join(repo_directory, "modules")
sys.path.append(modules_path)
import cv2

from models.Net import Net
from models.FS import FS
from utils.LoadData import *
from tqdm import tqdm

def parse_2d_list(input_string):
    try:
        # 將字符串解析為 Python 的內建數據結構
        result = ast.literal_eval(input_string)
        # 確保它是一個二維列表
        if all(isinstance(i, list) for i in result):
            return result
        else:
            raise argparse.ArgumentTypeError("Input is not a valid 2D list")
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Input is not a valid 2D list")

def addition(img):
    start_point_1 = (0, 1070)
    end_point_1 = (140, 1070)

    start_point_2 = (0, 1073)
    end_point_2 = (140, 1073)

    start_point_3 = (0, 1076)
    end_point_3 = (140, 1076)

    green_color = (0, 255, 0)  
    black_color = (0, 0, 0)
    thickness = 2

    cv2.line(img, start_point_1, end_point_1, green_color, thickness)
    cv2.line(img, start_point_2, end_point_2, black_color, thickness)
    cv2.line(img, start_point_3, end_point_3, green_color, thickness)

    return img

def generate_result_file(FS_posi, save_path, path, FRAME_COUNTER):

    file_path = os.path.join(save_path, path.split('/')[-1].split('.')[0]+"_"+str(FRAME_COUNTER)+".txt")
    np.savetxt(file_path, FS_posi, fmt='%d')

    # loaded_array = np.loadtxt(file_path, dtype=np.int64)


def main(opt):

    print('----------------------> Loading weights <----------------------')
    model = FS(weights = opt.weights, Backbone = opt.Backbone, bool_pretrained = False)

    img_pth = opt.source

    dataset = LoadImages(img_pth, img_size=opt.inference_size[0])
    
    vid_path, vid_writer = None, None

    os.makedirs('output', exist_ok=True)
    
    if not img_pth.endswith('.MP4'):
        save_path = os.path.join('./output', '{}.mp4'.format(img_pth.split('/')[-1].split(".")[0]))
    else:
        save_path = os.path.join('./output', img_pth.split('/')[-1])
        

    sub_pbar = tqdm(dataset, total=len(dataset))
    FRAME_COUNTER = 0
    for path, _img, im0s, vid_cap in sub_pbar:
        # im0s = addition(im0s)
        FRAME_COUNTER +=1
        try:
            FS_posi = model.predict(
                None, 
                im0s, 
                inference_size=opt.inference_size,
                panels=opt.panels, 
                filtration=True
                )
        except AttributeError:
            FS_posi = model.predict(
                im0s,
                inference_size=opt.inference_size,
                panels=opt.panels, 
                filtration=True
                )

        if opt.ShowFS:
            # Output the corresponding coordinates of each panel on the original screen.
            for sub_posi in FS_posi:
                cv2.polylines(im0s, [sub_posi], isClosed=False, color=(0, 255, 0), thickness=2)

                # Save FreeSpace posi to the .txt file
                if opt.Result_file_path:
                    generate_result_file(sub_posi, opt.Result_file_path, path, FRAME_COUNTER)

        # print(im0s.shape[1])
        # print(im0s.shape[0])
        # im0s = im0s[0:im0s.shape[0], 200:im0s.shape[1]]

        if opt.play_video:

            cv2.imshow('Out', im0s)
            keyResult = cv2.waitKey(1)
            
            if keyResult == ord('q'):
                cv2.destroyAllWindows()
                break
            elif keyResult == ord(' '):
                cv2.waitKey(0)

        if opt.save_video:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                if not img_pth.endswith('.MP4') and not img_pth.endswith('.mp4'):
                    fps = 30
                    w = im0s.shape[1]
                    h = im0s.shape[0]
                    vid_writer = cv2.VideoWriter(save_path.replace('.MP4', '_output.MP4').replace('.mp4', '_output.mp4').replace('.MOV', '_output.MOV'), cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                else:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = im0s.shape[1]
                    h = im0s.shape[0]
                    vid_writer = cv2.VideoWriter(save_path.replace('.MP4', '_output.MP4').replace('.mp4', '_output.mp4').replace('.MOV', '_output.MOV'), cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))

            vid_writer.write(im0s)

        # Add by Yan
        if opt.save_images:
            
            EvaluateDirName = "./output/panel"

            if not os.path.isdir(EvaluateDirName):
                os.makedirs(EvaluateDirName)

            if opt.save_image_amount == FRAME_COUNTER:
                break
            else:
                file_name = path.split('/')[-1].split('.')[0]+f'_{FRAME_COUNTER}.jpg'
                cv2.imwrite("{}/{}".format(EvaluateDirName, file_name),im0s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--source', type=str, default='/Default', help='Image or video for detecting.')
    parser.add_argument('--Backbone', type=str, default="MobileNetV3", help='Set backbone. Like MobileNetV3')
    parser.add_argument('--weights', type=str, default='/Default', help='The weights for model loading to detect.')
    parser.add_argument('--save_video', action='store_true', default=False, help='Store video')
    parser.add_argument('--play_video', action='store_true', default=False, help='Play video when detecting')
    parser.add_argument('--save_images', action='store_true', default=False, help='Store AI result images')
    parser.add_argument('--save_image_amount', type=int, default=None, help='Store AI result images amount')
    parser.add_argument('--SaveSingleImage', action='store_true', default=False, help='Save detect and original mix image or just save detect image.')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', type=str, default='0', help='GPU number')
    parser.add_argument('--Tusimple', action='store_true', default=False, help='Load Tusimple dataset')
    parser.add_argument('--inference_size', nargs='+', type=int, default=[512,288], help='Input size for model. ex.512 288 on command line')
    parser.add_argument('--panels', type=parse_2d_list, required=False, default='[["full"]]', help='recognize multiple regions on the original screen. ex.[[0,0,960,540] [960,0,1920,540]] on command line')
    parser.add_argument('--Result_file_path', type=str, default=None, help='Save result file on every frame.')
    parser.add_argument('--ShowFS', action='store_true', default=False, help='Like whether show FS line or not.')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(opt)
