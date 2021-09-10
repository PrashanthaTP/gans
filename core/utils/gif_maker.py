import glob
import imageio
import re
def create_gif(image_folder,gif_file,duration=0.5):
    with imageio.get_writer(gif_file, mode='I',duration=duration) as writer:
        filenames = glob.glob(image_folder+'//*.png' )
        filenames = sorted(filenames,key=lambda f:int(re.sub('\D','',f)))
      
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
