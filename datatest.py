from dataset import DDDataset

data = DDDataset('/projects/grail/jamesn8/projects/DepthRenderApproximator/output/texonly_hires', (512, 512), crop=True, insuffixes = ['_TEXONLY.exr'], masks='./mask', train=False, random_masks=True)

elem = data[0]

for item in elem:
    print(item.shape)
