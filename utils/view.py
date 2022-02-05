import math
import random

	

def display_mis_images(mis_images, mis_labels):
    len_images = len(mis_images) if len(mis_images)<=25 else 25
    num = math.ceil(math.sqrt(len_images))
    idxs = random.sample(range(0, len(mis_images)), len_images)
    
    fig = plt.figure(figsize=(num**2,num**2)) if num > 3 else plt.figure(figsize=((num+1)**2,(num+1)**2))
    for count, index in enumerate(idxs):
        ax = fig.add_subplot(num, num, count + 1, xticks=[], yticks=[])
        image = mis_images[count].reshape(28, 28)
        ax.set_title('Predict Label: {:d}'.format(mis_labels[count]), fontsize=24)
        ax.imshow(image)

        if count==len_images-1:
            break
            

def view_dir_images(imgs, type_image='png'):
    np.random.seed(42)  # wtf! the magic seed 
    fig, ax = plt.subplots()

    for img in imgs:
        ax.cla()
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([]) 
        # Note that using time.sleep does *not* work here!
        plt.pause(0.3)
