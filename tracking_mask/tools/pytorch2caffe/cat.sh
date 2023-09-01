#!/bin/bash
 
cat index.txt |while read line
do
cat /home/omnisky/programfiles/tracking/pysot/tools/cache/layer_with_weights/$line.prototxt >>model.prototxt
done
