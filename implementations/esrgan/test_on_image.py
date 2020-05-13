import math
from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("image_paths", metavar='image_paths', type=str, nargs='+', help="Path to images")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
parser.add_argument("--upscale_factor", type=int, default=2, help="Upscale factor")
opt = parser.parse_args()
print(opt)
UPSCALE_FACTOR=opt.upscale_factor
os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks,num_upsample=int(math.log2(UPSCALE_FACTOR))).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

for image_path in opt.image_paths:
    # Prepare input
    image_tensor = Variable(transform(Image.open(image_path))).to(device).unsqueeze(0)
    
    # Upsample image
    with torch.no_grad():
        sr_image = denormalize(generator(image_tensor)).cpu()

    # Save image
    fn = image_path.split("/")[-1]
    modn = os.path.basename(opt.checkpoint_model)
    modn = modn[:modn.rfind(".")]
    save_image(sr_image, f"/var/www/outputs/sr-{modn}-{fn}")
