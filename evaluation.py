import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize, gamma_correct, levels
from random import randint

def evaluate(model, dataset, device, filename, gamma=1, exposure=1, black=0.0, white=1.0, random=False):
    n = len(dataset)
    image, mask, gt = zip(*[dataset[randint(0, n-1) if random else i] for i in range(8)])
    #print('image shape: ',image[0].shape)
    if image[0].shape[0] == 4:
        use_guide = True
        guide = [im[3:4,:,:] for im in image]
        image = [im[:3,:,:] for im in image]
    else:
        use_guide = False
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    if use_guide:
        guide = torch.stack(guide)
    with torch.no_grad():
        if use_guide:
            output, _ = model(image.to(device), mask.to(device), guide.to(device))
        else:
            output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output
    

    grid = make_grid(
        torch.cat((levels(gamma_correct(unnormalize(image[:,0:3,:,:]), gamma, exposure), black, white), mask[:,0:3,:,:], levels(gamma_correct(unnormalize(output[:,0:3,:,:]), gamma, exposure), black, white),
                   levels(gamma_correct(unnormalize(output_comp[:,0:3,:,:]), gamma, exposure), black, white), levels(gamma_correct(unnormalize(gt[:,0:3,:,:]), gamma, exposure), black, white)), dim=0))
    save_image(grid, filename)
