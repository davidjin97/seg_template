import os

def write_data_txt(path,file_name):
    
    with open(file_name,'w') as f:
        # dirs [part1,...]
        for dirs in os.listdir(path):
            # dir [000001,...]
            for dir in os.listdir(os.path.join(path,dirs)):
                print(dir)
                temp = []
                for item in os.listdir(os.path.join(path,dirs,dir)):
                    file_path = os.path.join(path,dir,item)
                    print(file_path)
                    temp.append(file_path)
                data = temp[1]+" "+temp[3]+" "+temp[0]+" "+temp[2]+ "\n"
                f.write(data)
write_data_txt('./train','./InStereo2K_data.txt')

