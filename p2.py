import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

version = 'v3'
checkpoint_path = None

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def start_pred(image_path, out_path):

    def plot_att_map(ori, atts, h, w, caps):
        atts = atts.squeeze(0).detach().cpu().numpy()
        
        # <start> + cap + <end>
        fig_w, fig_h = 5, (len(caps)+2)//5+2

        axes = []
        fig=plt.figure(figsize=(fig_w*3,fig_h*3))

        axes.append( fig.add_subplot(fig_w, fig_h, 1) )
        axes[-1].set_title('<start>')
        plt.axis('off')
        plt.imshow(ori)


        caps.append('<end>')
        for i in range(len(caps)):

            att_ = np.reshape(atts[i],(h, w))
            mask = cv2.resize(att_, (img_w,img_h))
            normed_mask = mask / mask.max()
            normed_mask = (normed_mask * 255).astype('uint8')


            axes.append( fig.add_subplot(fig_w, fig_h, i+2) )
            axes[-1].set_title(caps[i])
            plt.axis('off')
            plt.imshow(ori, alpha=1)
            plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap="jet")
        plt.savefig(os.path.join(out_path,os.path.splitext(os.path.basename(image_path))[0]+'.png'))
        
            
    def create_caption_and_mask(start_token, max_length):
        caption_template = torch.zeros((1, max_length), dtype=torch.long)
        mask_template = torch.ones((1, max_length), dtype=torch.bool)

        caption_template[:, 0] = start_token
        mask_template[:, 0] = False

        return caption_template, mask_template


    @torch.no_grad()
    def evaluate():
        out_word_amt = 0
        model.eval()
        for i in range(config.max_position_embeddings - 1):
            predictions, attn, h, w = model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                return caption, attn, h, w

            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False
            out_word_amt += 1
        
        return caption, attn, h, w

    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

    image = Image.open(image_path)
    img = image
    img_h, img_w = image.height, image.width
    image = coco.val_transform(image)
    image = image.unsqueeze(0)


    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)
    output, attn, h, w = evaluate()
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(result.capitalize())
    plot_att_map(img, attn, h, w, result.capitalize().split())



parser =  argparse.ArgumentParser(description='Use to caption for HW3_p2')
parser.add_argument( '--input_path', type=str, default='', help='path to testing images in the target domain' )
parser.add_argument( '--output_path', type=str, default='', help='path to your output prediction file')
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

imgs_path = glob.glob(os.path.join(input_path,'*'))
# print(imgs_path)
for p in imgs_path:
    start_pred(p,output_path)