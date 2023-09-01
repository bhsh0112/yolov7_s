
**注意**！！
config 和Weight需要用最新的！！！

CUDA_VISIBLE_DEVICES=1 python tools/demo.py --config /home/omnisky/programfiles/tracking/pysot/files/config.yaml --snapshot /home/omnisky/programfiles/tracking/pysot/files/model.pth --video_name /home/omnisky/programfiles/robint/charging/data/modify/

python tools/demo.py --config experiments/siammaske_r50_l3/config.yaml --snapshot experiments/siammaske_r50_l3/model.pth --video_name 1.avi
