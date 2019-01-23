from dataset import DDDataset

data = DDDataset('/projects/grail/jamesn8/projects/DepthRenderApproximator/output/texonly_hires', (512, 512), crop=True, insuffixes = ['_TEXONLY.exr'], masks=[('/local1/edzhang/dataset/masks', '_objectmask.png')], train=False, random_masks=False, auto_resize=True, depth_map=('/projects/grail/edzhang/differential/data','_WO.exr'))

elem = data[0]

for item in elem:
    print(item.shape)
#for i, item in enumerate(elem):
#    print('element {}: '.format(i), item)
