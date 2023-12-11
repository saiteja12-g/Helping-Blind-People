"""
    Small script for testing on few generic images given the model weights.
    In order to minimize the requirements, it runs only on CPU and images are
    processed one by one.
"""
import requests
import torch
import argparse
import pickle
from argparse import Namespace
import cv2
import numpy as np
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description
from utils.image_path import get_image_path
import threading
import time
import torchvision
from PIL import Image as PIL_Image

global frame
frame = None

def get_live_feed():
    global frame
    # link to camera feed from camera
    url = "http://192.168.1.157:8080/shot.jpg"
    # cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    # Check if the camera opened successfully
    # if not cap.isOpened():
    #     print("Error: Could not open camera.")
    #     return

    i = 0
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        # Capture frame-by-frame
        # ret, new_frame = cap.read()
        
        img = cv2.imdecode(img_arr, -1) 
        transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((384, 384))])
        transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
        pil_image = PIL_Image.fromarray(img)
        preprocess_pil_image = transf_1(pil_image)
        image = torchvision.transforms.ToTensor()(preprocess_pil_image)
        image = transf_2(image)
        # img = image.unsqueeze(0)
        frame = image.numpy()
        # frame = image.clone()
        print(frame.shape)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Display the resulting frame
        cv2.imshow('Live Video Feed', frame)
        # print(i)
        # if i%100==0:
        #     cv2.imwrite(f'frames/frame{i}.png', frame)

        # Press 'q' on keyboard to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i+=1

    # When everything done, release the video capture object
    # cap.release()
    cv2.destroyAllWindows()
    frame = None

def main():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--max_seq_len', type=int, default=74)
    parser.add_argument('--load_path', type=str, default='./demo_material/rf_model.pth')
    parser.add_argument('--video_path', type=str, default='./demo_material/video_2.mp4')
    parser.add_argument('--beam_size', type=int, default=5)

    args = parser.parse_args()

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model_args = Namespace(model_dim=args.model_dim,
                           N_enc=args.N_enc,
                           N_dec=args.N_dec,
                           dropout=0.0,
                           drop_args=drop_args)

    with open('./demo_material/demo_coco_tokens.pickle', 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    print("Dictionary loaded ...")

    threading.Thread(target=get_live_feed, daemon=True).start()

    img_size = 384
    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=args.max_seq_len, drop_args=model_args.drop_args,
                                rank='cpu')
    checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded ...")

    global frame

    i = 0
    while frame is not None:
        tensor_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float()
        tensor_frame = tensor_frame.permute(2, 0, 1)  # Convert HWC to CHW
        tensor_frame = tensor_frame.unsqueeze(0)  # Add a batch dimension

        if i%20==0:
            beam_search_kwargs = {'beam_size': args.beam_size,
                                'beam_max_seq_len': args.max_seq_len,
                                'sample_or_max': 'max',
                                'how_many_outputs': 1,
                                'sos_idx': sos_idx,
                                'eos_idx': eos_idx}
            with torch.no_grad():
                pred, _ = model(enc_x=tensor_frame,
                                enc_x_num_pads=[0],
                                mode='beam_search', **beam_search_kwargs)
            pred = tokens2description(pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)
            print('Description: ' + pred + '\n')
        i+=1
        time.sleep(0.1)

    print("Closed.")


if __name__ == "__main__":
    main()