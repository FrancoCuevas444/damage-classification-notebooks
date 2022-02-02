import dataset_modules.common as common
import torchvision.transforms as transforms
import time
from sys import getsizeof

def preload_images(
        images_folder='./dataset_modules/imgs/', 
        state_file='./dataset_modules/state.json', 
        resize_to=224):
    
    metadata = common.load_metadata_dataframe(state_file, filter_useful=True)
    
    start = time.time()
    print("Started to preload images...")
    
    result = dict(metadata.apply(lambda x: preload_image(images_folder, x['image'], resize_to), axis=1).tolist())
    
    size_in_bytes = getsizeof(result)
    time_elapsed = time.time() - start
    
    print('Image preloading complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('The preloaded images use {}MB of memory'.format(size_in_bytes / 1024 / 1024))
    
    return result
    
def preload_image(images_folder, img_name, resize_to):
    img_path = images_folder + img_name
    preloaded_image = common.pil_loader(img_path)
    
    if resize_to:
        transform = transforms.Resize((resize_to, resize_to))
        preloaded_image = transform(preloaded_image)
    
    return (img_name, preloaded_image)