import glob,os,time
from PIL import Image

image_folder=r"D:\cli\gifs" # drop images in folder, set path here
file_pattern="*.png" # image type (jpg,png,webp,etc)
frame_duration=90 # duration/ms per frame (100=10fps,200=5fps)

frames,ti,search_pattern=[],time.time(),os.path.join(image_folder, file_pattern)
image_files=sorted(glob.glob(search_pattern))
if not image_files:print(f"Not found '{search_pattern}'.")
else:print(f"Found {len(image_files)} images. creating gif.")
for img_path in image_files:
    try:img = Image.open(img_path);frames.append(img);print(f" {len(frames)}/{len(image_files)} {os.path.basename(img_path)}")
    except Exception as e:print(f" -- {img_path}. Error: {e}")
if frames:frames[0].save(f"{str(ti)[:10]}"+"_vid.gif",format='GIF',append_images=frames[1:],save_all=True,duration=frame_duration,loop=0);print(f"\n{len(image_files)} images in {time.time()-ti:.2f} secs")
else:print(f"No valid image frames")
