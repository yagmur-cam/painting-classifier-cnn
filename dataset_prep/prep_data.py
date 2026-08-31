import os
import shutil 
import random 
import requests
from PIL import Image
from PIL import ImageOps
import imagehash

random.seed(42)

#Split all 7 styles into train/val/test
BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.environ.get("PAINTING_SOURCE", os.path.join(BASE, "raw"))
DESTINATION = os.path.join(BASE, "..", "dataset")
SURREALISM_CSV = os.path.join(BASE, "surrealism.csv")
SURREALISM_OUTPUT = os.path.join(SOURCE, "Surrealism") #where the Surrealism folder is 

STYLES = [
    "Baroque",
    "Cubism",
    "Expressionism",
    "Impressionism",
    "Romanticism",
    "Surrealism",
    "Renaissance"
]

#inside the SOURCE folder only look for these folders and ignore the rest
#we have 3 types of Renaissance in SOURCE also doesnt have Surrealism
RENAISSANCE_FOLDERS = ["Early_Renaissance", "High_Renaissance", "Northern_Renaissance"]
SPLIT = {"train": 0.70, "validation": 0.15, "test": 0.15}

#-----------------------------------------------------------------------------------------------------------------------

def download_surrealism():
    os.makedirs(SURREALISM_OUTPUT, exist_ok=True) 
    with open(SURREALISM_CSV, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            if len(parts) < 5: #one line is 5 parts
                continue
            url = parts[-1].strip() 
            filename = f"surrealism_{i}.jpg"
            filepath = os.path.join(SURREALISM_OUTPUT, filename)
            try: 
                response = requests.get(url, timeout=10) 
                if response.status_code  == 200: 
                    with open(filepath, "wb") as img: 
                        img.write(response.content) 
                    print(f"Downloaded {i}: {filename}")
                else:
                    print(f"Skipped {i}: bad status {response.status_code}")
            except Exception as e: 
                print(f"Failed {i}: {e}")


#--------------------------------------------------------------------------------------------------------------------------------


def merge_renaissance(): 
    renaissance_output = os.path.join(SOURCE, "Renaissance")
    os.makedirs(renaissance_output, exist_ok=True) #create the output folder for Renaissance

    for folder in RENAISSANCE_FOLDERS: 
        folder_path = os.path.join(SOURCE, folder) #join this Renaissance folder to the main SOURCE file
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder}")
            continue 
        images = os.listdir(folder_path)
        for image in images:
            src = os.path.join(folder_path, image)
            dst = os.path.join(renaissance_output, f"{folder}_{image}") 
            shutil.copy(src, dst) 
        print(f"Merged {len(images)} images from {folder}")

    print(f"Renaissance merge complete")


#---------------------------------------------------------------------------------------------------------------------------------------------------------

def clean_dataset():
    for style in STYLES:
        style_path = os.path.join(SOURCE, style)
        if not os.path.exists(style_path):
            print(f"Folder not found: {style_path}")
            continue 
        images = os.listdir(style_path)
        removed = 0
        for image in images:
            filepath = os.path.join(style_path, image)
            if not image.lower().endswith((".jpg", ".jpeg", ".png")):
                os.remove(filepath)
                removed += 1
                continue 
            if os.path.getsize(filepath) < 5000:
                os.remove(filepath)
                removed += 1
                continue
            if os.path.getsize(filepath) > 5000000:
                os.remove(filepath)
                removed += 1
                continue
            try:
                img = Image.open(filepath)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    img.save(filepath)
            except:
                os.remove(filepath)
                removed += 1
                continue
        #remove duplicates using file hashing
        seen_hashes = set() 
        for image in os.listdir(style_path):
            filepath = os.path.join(style_path, image)
            try:
                img = Image.open(filepath)
                file_hash = imagehash.phash(img)
                if any(abs(file_hash - seen) < 10 for seen in seen_hashes):
                    os.remove(filepath)
                    removed += 1
                else:
                    seen_hashes.add(file_hash)
            except:
                os.remove(filepath)
                removed += 1
        print(f"{style}: removed {removed} bad files, kept {len(os.listdir(style_path))} images")

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def balance_dataset(target=2500):
    for style in STYLES:
        style_path = os.path.join(SOURCE, style)
        if not os.path.exists(style_path):
            print(f"Folder not found: {style_path}")
            continue
        images = [f for f in os.listdir(style_path)
                  if f.lower().endswith((".jpg", ".jpeg", "png"))]
        total = len(images)
        if total > target: 
            random.shuffle(images) #shuffles first so the removal is random, not just deleting the last files alphabetically
            images_to_remove = images[target:] #remove all the images after the target is reached
            for image in images_to_remove:
                os.remove(os.path.join(style_path, image))
            print(f"{style}: reduced from {total} to {target} images")
        elif total < target:
            print(f"{style}: only {total} images, below target of {target} -- add more manually ")
        else:
            print(f"{style}: already at {target} images")


#---------------------------------------------------------------------------------------------------------------------------------------------------------------


def pre_process_images():
    target_size = (128, 128) 
    for split in ["train", "validation", "test"]:
        for style in STYLES:
            style_path = os.path.join(SOURCE, split, style)
            if not os.path.exists(style_path):
                print(f"Folder not found: {style_path}")
                continue
            images = [f for f in os.listdir(style_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for image in images:
                filepath = os.path.join(style_path, image)
                try:
                    img = Image.open(filepath).convert("RGB") 
                    new_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    new_img.save(filepath)
                except Exception as e:
                    print(f"Failed {image}: {e}")
            print(f"{style}: resized {len(images)} images to 128x128")


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


def split_dataset():
    for style in STYLES:
        style_path = os.path.join(SOURCE, style)
        if not os.path.exists(style_path):
            print(f"Folder not found: {style_path}")
            continue 
        images = [f for f in os.listdir(style_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))] 
        random.shuffle(images) 
        total = len(images)

        #calculate the cutoff points for 70/15/15 
        train_end = int(total * SPLIT["train"])
        val_end = train_end + int(total * SPLIT["validation"])

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for split_name, split_images in [("train", train_images),
                                         ("validation", val_images),
                                         ("test", test_images)]:
            split_dir = os.path.join(DESTINATION, split_name, style)
            os.makedirs(split_dir, exist_ok=True)  
            for image in split_images:
                src = os.path.join(style_path, image)
                dst = os.path.join(split_dir, image)
                shutil.copy(src, dst) 
            print(f"{style} --> {split_name}: {len(split_images)} images")
        print(f"{style} split complete. Total: {total}")


if __name__ == "__main__": 
    download_surrealism() # ---> there wasnt a surrealism foder in the main data resource
    merge_renaissance() # ---> there were 3 different folders for Renaissance
    clean_dataset() 
    balance_dataset() 
    pre_process_images()  
    split_dataset() 