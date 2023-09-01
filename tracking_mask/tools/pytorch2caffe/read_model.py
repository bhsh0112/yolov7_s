import numpy as np
import caffe
import torch
# from resnet_18 import resnet18_fc512
import os
import re
import linecache

#### extract and save pytorch weights as blobs
def save_pytorch_weights(model, pytorch_weights_dir):

    for name, param in model.named_parameters():
        print('-----layer >>>>>', name)
        weight = param.detach().numpy()
        print(name, weight.shape)
        fname = pytorch_weights_dir + '/' + name + '.prototxt'
        if not os.path.exists(fname):
            os.mknod(fname)
        f = open(fname, 'w')
        
        ## different layers
        if weight.ndim == 4:   # convolutional weight   shape = [out_c, input_c, h, w]
            weight_shape = weight.shape
            weight_1d = weight.reshape(weight.shape[0] * weight.shape[1] * weight.shape[2] * weight.shape[3])
        elif weight.ndim == 2:  # fc weight   shape = [out, input]
            weight_shape = weight.shape  
            weight_1d = weight.reshape(weight.shape[0] * weight.shape[1])
        elif weight.ndim == 1:   # bias 
            weight_shape = weight.shape
            weight_1d = weight

        f.write('  blobs {\n')
        for v in weight_1d:
            f.write('    data: %8f' % v)
            f.write('\n')
        f.write('    shape {\n')
        for s in weight_shape:
            f.write('      dim: ' + str(s)) #print dims
            f.write('\n')
        f.write('    }\n')
        f.write('  }\n')
        f.write('}\n')

        f.close()


#### extract pytorch Batchnorm weights, then split into caffe bn and scale , save blobs
def extract_bn_weights(pre_dict):
    # resnet18 = resnet18_fc512()    
    # checkpoint = torch.load("/home/yao/Desktop/test_output/reid/pytorch/checkpoint_ep99.pth.tar")
   
    # pre_dict = remove_fc(checkpoint)
    
    # resnet18.load_state_dict( pre_dict['state_dict'] )
    # resnet18.eval()

    # print(resnet18.conv1.weight[0,0,0,:])
    # print(resnet18)
    
    # for k in resnet18.state_dict():
    #     print("Layer {}".format(k))
    #print(resnet18.bn1.running_mean.cpu().numpy())
    #print(resnet18.bn1.running_var.cpu().numpy())
    #print(resnet18.conv1.weight.detach().numpy()[0,0,0,:])
    # print(resnet18.bn1.bias.detach().numpy())
    for k,weight in pre_dict.items():
        if 'running_mean' in k:
            print(k)
            fname = os.path.join('/home/omnisky/programfiles/tracking/pysot/tools/cache', k+'.prototxt')
            if not os.path.exists(fname):
                os.mknod(fname)
            f = open(fname, 'w')
            weight = weight.cpu().numpy()
            assert weight.ndim == 1
            weight_shape = weight.shape
            f.write('  blobs {\n')
            for v in weight:
                f.write('    data: %8f' % v)
                f.write('\n')
            f.write('    shape {\n')
            for s in weight_shape:
                f.write('      dim: ' + str(s)) #print dims
                f.write('\n')
            f.write('    }\n')
            f.write('  }\n')
            f.write('}\n')

            f.close()
        if 'running_var' in k:
            print(k)
            fname = os.path.join('/home/omnisky/programfiles/tracking/pysot/tools/cache', k+'.prototxt')
            if not os.path.exists(fname):
                os.mknod(fname)
            f = open(fname, 'w')
            weight = weight.cpu().numpy()
            assert weight.ndim == 1
            weight_shape = weight.shape
            f.write('  blobs {\n')
            for v in weight:
                f.write('    data: %8f' % v)
                f.write('\n')
            f.write('    shape {\n')
            for s in weight_shape:
                f.write('      dim: ' + str(s)) #print dims
                f.write('\n')
            f.write('    }\n')
            f.write('  }\n')
            f.write('  blobs {\n')
            f.write('    data: 1\n')
            f.write('    shape {\n')
            f.write('      dim: 1\n')
            f.write('    }\n')
            f.write('  }\n')
            f.write('}\n')

            f.close()

### split caffe layers and save it as its 'name'
def split_layers(layer_prototxt, layers_dir):
    f = open(layer_prototxt)
    keyword = 'layer {'
    line_numbers = []
    num = 1
    
    for line in f.readlines():
        m = re.search(keyword,line)
        if m is not None:
            line_numbers.append(num)
        num += 1
    size = len(line_numbers)
    
    total = []
    for index, _ in enumerate(line_numbers):
        ## split by 'layer'
        start = line_numbers[index]
        if index != (size-1):
            end = line_numbers[(index+1)]
            destlines = linecache.getlines(layer_prototxt)[start-1 : end-1]
        else:
            destlines = linecache.getlines(layer_prototxt)[start-1 : ]
        
        total.append(destlines)
   
    for layer in total:
        for i in layer:
            if 'name' in i:                   
                name_ = i.strip().split('"')
                name = name_[1]
                print(name)
                layer_prototxt = layers_dir + '/' + name + '.prototxt'
                layer_f = open(layer_prototxt, 'w')
                for ii in layer:
                    layer_f.write(ii)
                layer_f.close()



def main():
    save_weights_dir = '/home/yao/Desktop/test_output/caffemodel/save_weights_new'
    layers_dir = '/home/omnisky/programfiles/tracking/pysot/tools/cache/layers/'
    layer_prototxt = '/home/omnisky/programfiles/tracking/Robint_SiameseRPN/model/split.prototxt'

  
    ## extract pytorch weights of every layer and save into prototxt in <pytorch_weights_dir> 
 
    #save_pytorch_weights(save_weights_dir)

    #split_layers(layer_prototxt, layers_dir)
    
    # extract_bn_weights()


if  __name__ == "__main__":
   main() 
