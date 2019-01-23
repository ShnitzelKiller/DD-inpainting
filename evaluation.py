import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
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
        torch.cat((unnormalize(image[:,0:3,:,:]), mask[:,0:3,:,:], unnormalize(output[:,0:3,:,:]),
                   unnormalize(output_comp[:,0:3,:,:]), unnormalize(gt[:,0:3,:,:])), dim=0))
    save_image(grid, filename)
