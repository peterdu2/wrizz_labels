import argparse
import pandas as pd
import cv2
import math
import random
from pathlib import Path


ADVANCE = 119
DEVANCE = 115
LEFT = 97
RIGHT = 100
SAVE_QUIT = 32
FORCE_QUIT = 27


def draw_cross(img, x, y, w, h, fraction=0.04, thickness=4, color=(0,0,255)):
    l1A = (int(x - w*fraction), int(y))
    l1B = (int(x + w*fraction), int(y))
    l2A = (int(x), int(y - w*fraction))
    l2B = (int(x), int(y + w*fraction))
    cv2.line(img, l1A, l1B, color, thickness=thickness)
    cv2.line(img, l2A, l2B, color, thickness=thickness)


def to_anno_string(annos):
    if len(annos) == 0:
        return ""
    annostr = ""
    for a in annos:
        annostr += "[{},{},{},{},{}];".format(
            *[int(x) for x in a]
        )
    return annostr[:-1]


def from_anno_string(annostr):
    if len(annostr) == 0:
        return []
    split = annostr.split(';')
    annos = []
    for s in split:
        annos.append([int(x) for x in s[1:-1].split(',')])
    return annos


def label_image(image_file, csv, nannos):
    print("labeling {}".format(image_file.name))
    img = cv2.imread(str(image_file))
    h, w = img.shape[0], img.shape[1]
    print("resolution = {} x {}".format(w, h))
    if csv['image_name'].eq(image_file.name).any():
        annotations = from_anno_string(csv.iloc[(csv['image_name'] == image_file.name).argmax()]['labels'])
    else:
        annotations = []
    
    return_code = None
    current_anno = None
    idx = 0
    while idx < nannos:
        if current_anno is None:
            if idx < len(annotations):
                x1, y1, x2, y2, l = annotations[idx]
            else:
                x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
                x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
                while math.sqrt((x1 - x2)**2 + (y1 - y2)**2) < min(h*0.05, w*0.05):
                    # must have labeled pairs be at least 5% of the image away from other pair
                    x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
                l = -1
        else:
            x1, y1, x2, y2, l = current_anno

        # render the annotation
        render_img = img.copy()
        if l == 0: # 0 => same label
            draw_cross(render_img, x1, y1, w, h, color=(255,0,0))
            draw_cross(render_img, x2, y2, w, h, color=(255,0,0))
        elif l == 1: # 1 => second is more traversable
            draw_cross(render_img, x1, y1, w, h, color=(0,0,255))
            draw_cross(render_img, x2, y2, w, h, color=(0,255,0))
        else: # -1 => first is more traversable
            draw_cross(render_img, x1, y1, w, h, color=(0,255,0))
            draw_cross(render_img, x2, y2, w, h, color=(0,0,255))
        
        cv2.imshow('window', render_img) 
        while cv2.getWindowProperty('window', cv2.WND_PROP_VISIBLE) > 0:
            key_code = cv2.waitKey(100)
            if key_code in [ADVANCE, DEVANCE, LEFT, RIGHT, SAVE_QUIT, FORCE_QUIT]:
                break

        if key_code == ADVANCE: # TODO: add key code for up arrow.
            if idx < len(annotations):
                annotations[idx] = [x1, y1, x2, y2, l]
            else:                
                annotations.append([x1, y1, x2, y2, l])
            idx = idx + 1
            current_anno = None
            if idx >= nannos:
                return_code = 1
                break
        elif key_code == DEVANCE: # TODO: add key code for down arrow
            if idx < len(annotations):
                annotations[idx] = [x1, y1, x2, y2, l]
            else:
                annotations.append([x1, y1, x2, y2, l])
            idx = idx - 1
            current_anno = None
            if idx < 0:
                return_code = -1
                break
        elif key_code == LEFT: # TODO: add key code for left arrow
            current_anno = [x1, y1, x2, y2, max(l-1, -1)]
        elif key_code == RIGHT: # TODO: add key code for right arrow
            current_anno = [x1, y1, x2, y2, min(l+1,  1)]
        elif key_code == FORCE_QUIT: # escape doesn't save
            return_code = 0
            break
        elif key_code == SAVE_QUIT: # space saves and exits
            return_code = 0
            if idx < len(annotations):
                annotations[idx] = [x1, y1, x2, y2, l]
            else:                
                annotations.append([x1, y1, x2, y2, l])
            break
    
    if len(annotations) > 0:
        row_num = None
        if csv['image_name'].eq(image_file.name).any():
            row_num = (csv['image_name'] == image_file.name).argmax()
        dictionary = {c: list(csv[c]) for c in csv.columns}
        if row_num is None:
            dictionary['image_name'].append(image_file.name)
            dictionary['width'].append(w)
            dictionary['height'].append(h)
            dictionary['labels'].append(to_anno_string(annotations))
        else:
            assert(dictionary['image_name'][row_num] == image_file.name)
            dictionary['width'][row_num] = w
            dictionary['height'][row_num] = h
            dictionary['labels'][row_num] = to_anno_string(annotations)
        csv = pd.DataFrame(dictionary)
        csv = csv.sort_values(by='image_name')
    return return_code, csv


def main(args):
    print("Launching labeler...")
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError("Directory does not exist or is invalid")

    csv_file = Path(args.csv)
    filetype = args.filetype
    if args.annos > 0:
        annos = args.annos
    else:
        raise ValueError("Need at least 1 annotation per image")
    
    if csv_file.exists():
        if csv_file.is_dir():
            raise ValueError("CSV file is invalid")
        else:
            csv = pd.read_csv(csv_file)
            if csv.shape[0] > 0:
                if (csv.columns == ['image_name', 'width', 'height', 'labels']).min() != True:
                    raise ValueError("Invalid existing CSV")
            else:
                csv = pd.DataFrame({'image_name': [], 'width': [], 'height': [], 'labels': []})
    else:
        csv = pd.DataFrame({'image_name': [], 'width': [], 'height': [], 'labels': []})

    image_files = sorted(list(directory.glob('*.{}'.format(filetype))))
    
    if len(image_files) < 1:
        print("No files found to label")
    else:
        idx = 0
        if args.skip_done:
            while csv['image_name'].eq(image_files[idx].name).any():
                idx = min(idx+1, len(image_files)-1)
                print(idx)
        print(idx)

        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("window", 320*3, 240*3)
        while True:
            image_file = image_files[idx]
            print("Total number of labeled files = {}".format(csv.shape[0]))
            result_code, new_csv = label_image(image_file, csv, nannos=annos)
            csv = new_csv
            csv.to_csv(csv_file, index=False)
            if result_code == -1:
                idx = max(idx-1, 0)
            elif result_code == 1:
                idx = min(idx+1, len(image_files)-1)
            else:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Girish's dataset
    #python labeler.py ./../data/WayFAST/imgs ./../data/WayFAST/labels.csv --filetype tif --annos 1 
    # CaT
    #python labeler.py ./../data/CAT/train/imgs ./../data/CAT/train/labels.csv --filetype png --annos 1
    #python labeler.py ./../data/CAT/test/imgs ./../data/CAT/test/labels.csv --filetype png --annos 1
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("csv", type=str)
    parser.add_argument("--filetype", type=str, default="png") # file type for the images
    parser.add_argument("--annos", type=int, default=2) # number of annoations per image
    parser.add_argument("--skip_done", type=int, default=1) # if 0, we do not skip unlabeled
    args = parser.parse_args()
    main(args=args)
