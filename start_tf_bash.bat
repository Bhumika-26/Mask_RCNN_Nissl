#! /bin/bash
docker run -it --ipc=host --gpus=all -p 9999:9999 -v c:/users/keerthi/code/hbp/samik_code/bhumika_maskrcnn_nissl:/workspace 73be11373498 bash