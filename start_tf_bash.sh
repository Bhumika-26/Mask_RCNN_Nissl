#! /bin/bash
docker run -it --ipc=host --gpus=all -p 9999:9999 -v /home/bhumika/Img_segmentation:/workspace 73be11373498 bash