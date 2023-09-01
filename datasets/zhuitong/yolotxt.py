'''
/*************************************************
*
**Description：yolo训练时，遍历训练集与验证集中文件中的图片路径，并写入到指定的txt文件中

** Author：慕灵阁-wupke
** Time：2022-01-06
** Versions ：遍历单一文件夹图片中的路径，写入到指定的txt文件中
**
*
***************************************************/'''
import os
import os.path

if __name__ == "__main__":

    pic_path = 'images/train/'   # 要遍历的图片文件夹路径
    save_txtfile = open('train.txt','w') # 保存路径的记事本文件

    #pic_path = 'images/val/'   # 要遍历的图片文件夹路径
    #save_txtfile = open('valid.txt','w') # 保存路径的记事本文件


    # i = 0
    for root, dirs, files in os.walk(pic_path):
# root 所指的是当前正在遍历的这个文件夹的本身的地址
# dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
# files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
        for file in files:
            print(os.path.join(root,file))
            # save_txtfile.write(os.path.join(root,file) + ' ' + str(i) +'\n')
            save_txtfile.write(os.path.join(root,file) +'\n')
    print('The files path of ' + str(pic_path) + 'has already written to' + str(save_txtfile) )
    save_txtfile.close();
