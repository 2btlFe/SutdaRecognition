import os
import ipdb

if __name__ == "__main__":

    dataset = "/workspace/OD_dataset/label"
    for root, dir, files in os.walk(dataset):
        
        #ipdb.set_trace()

        for file in files:
            
            label = os.path.basename(root)
            
            new_txt = file[:-4] + ".txt" #.jpg 제거
            new_file = os.path.join(root, new_txt)

            with open(new_file, "w") as f:
                f.write(label)

            