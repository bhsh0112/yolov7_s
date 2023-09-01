# 模型转换 pytorch >> caffe

## -----step1------

将pytorch model中的参数提取出来，每一层分别存成一个prototxt文件，按照层名命名

*save_pytorch_weights* 函数

## -----step2------

整理caffe model 的网络参数文件（对应有pytorch参数的层都调整了 } 的对应，以对应合并后的完整性），然后每一层都分别存成一个prototxt，（本次以网络文件中的 ‘*name*’ 命名）。

 *split_layers* 函数

ps: poolimg 层 在caffe源码里将 取整方式由ceil 改为floor，保证输出尺寸统一

## -----step3------

整理一个index.txt文件，按照层 prototxt 和带参数的prototxt 顺序排列。

**examples:**

<u>conv1.prototxt</u>
<u>conv1.weight.prototxt</u>
<u>bn1.prototxt</u>
<u>bn1.running_mean.prototxt</u>
<u>bn1.running_var.prototxt</u>
<u>scale1.prototxt</u>
<u>bn1.weight.prototxt</u>
<u>bn1.bias.prototxt</u>
<u>relu1.prototxt</u>
<u>pool1.prototxt</u>
<u>layer1.0.conv1.prototxt</u>
<u>layer1.0.conv1.weight.prototxt</u>
<u>layer1.0.bn1.prototxt</u>
<u>layer1.0.bn1.running_mean.prototxt</u>
<u>layer1.0.bn1.running_var.prototxt</u>
<u>layer1.0.scale1.prototxt</u>
<u>layer1.0.bn1.weight.prototxt</u>
<u>layer1.0.bn1.bias.prototxt</u>
<u>layer1.0.relu1.prototxt</u>

......

**BN层互相转换：**

**batchnorm 层 >>**

`bn.running_mean = bn.blobs[0].data / blobs[2].data[0]`

`bn.running_var  =  bn.blobs[0].data / blobs[2].data[0]`

`blobs[2] :   blobs {`
                       `data: 1`
                       `shape {`
                            `dim: 1`
                            `}`
                    `}`

**scale 层 >>**

`bn.weight = blobs[0].data`

`bn.bias = blobs[1].data`

## -----step4------

将需要合并的所有prototxt汇集，写一个cat文件，合并成一个大的prototxt，里面有网络文件和参数。

cat.sh :

`#!/bin/bash`

`cat index.txt |while read line`
`do`
`cat $line >>model.prototxt`
`done`

## -----step5------

将prototxt转成二进制caffemodel

*wm.cpp*

CMakeLists.txt

`cmake .`

`make all`

`./write_model`

## -----step6------

修改 model.prototxt 中一些小错误: 

**1.检查对应参数是否齐全， 与括号的对应**

**2.Check failed: target_blobs.size() == source_layer.blobs_size() (1 vs. 0) Incompatible number of blobs for layer**

参数的维度不对应（如bn 层有三组参数，scale层有两组，参数的维度要对应）

**3.检查对应层的参数**

**<u>pytorch hook:</u>**

`fmlist = []`
`def hook(module, inputdata, output):`    

​      `fmlist.append(output.data.cpu().numpy())`

`handle = resnet18.fc[2].register_forward_hook(hook)`

`fc = resnet18(data)`

`handle.remove()`

`print(fmlist)`

**<u>caffe:</u>**

`print(net.forward(end='conv1'))`

**4.load pytorch model 删除部分层后注意参数是否正确载入**

