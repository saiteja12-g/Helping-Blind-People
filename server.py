# Import necessary libraries
from flask import Flask, render_template, Response
import torch
import cv2
import threading
import time
import argparse
import pickle
from argparse import Namespace
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description

app = Flask(__name__)

# Global variables
global frame
global model
global sos_idx
global eos_idx
global description


# Initialize variables
frame = None
description = "Waiting for frame..."

def get_live_feed():
    global frame
    global description

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    # Check if the camera opened successfully
    if not cap.isOpened():
        description = "Error: Could not open camera."
        return

    i = 0
    while True:
        # Capture frame-by-frame
        ret, new_frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            description = "Can't receive frame (stream end?). Exiting ..."
            break

        frame = cv2.resize(new_frame, (384, 384))

        # Process frame and generate description at regular intervals
        if i % 20 == 0:
            tensor_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float()
            tensor_frame = tensor_frame.permute(2, 0, 1)  # Convert HWC to CHW
            tensor_frame = tensor_frame.unsqueeze(0)  # Add a batch dimension

            beam_search_kwargs = {
                'beam_size': args.beam_size,
                'beam_max_seq_len': args.max_seq_len,
                'sample_or_max': 'max',
                'how_many_outputs': 1,
                'sos_idx': sos_idx,
                'eos_idx': eos_idx
            }
            with torch.no_grad():
                pred, _ = model(enc_x=tensor_frame, enc_x_num_pads=[0], mode='beam_search', **beam_search_kwargs)
            pred = tokens2description(pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)
            description = 'Description: ' + pred

        i += 1
        time.sleep(0.1)

    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    frame = None

def init_model():
    global model
    global sos_idx
    global eos_idx
    global coco_tokens

    # Load model configuration
    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model_args = Namespace(model_dim=args.model_dim, N_enc=args.N_enc, N_dec=args.N_dec, dropout=0.0, drop_args=drop_args)

    # Load dictionary
    with open('./demo_material/demo_coco_tokens.pickle', 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    # Load model
    img_size = 384
    model = End_ExpansionNet_v2(
        swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
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
        rank='cpu'
    )
    
    # Load model weights
    # checkpoint = torch.load(args.load_path)
    checkpoint = torch.load(args.load_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

@app.route('/')
def index():
    return render_template('index.html', description=description)

def generate():
    while True:
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Web Demo')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--max_seq_len', type=int, default=74)
    parser.add_argument('--load_path', type=str, default='./demo_material/rf_model.pth')
    parser.add_argument('--beam_size', type=int, default=5)
    args = parser.parse_args()

    # Initialize model and start video feed thread
    init_model()
    threading.Thread(target=get_live_feed, daemon=True).start()

    # Run Flask app
    app.run(debug=True, threaded=True)
