#!/bin/bash
printf "running extract_britain_elects.py\n"
/home/ubuntu/anaconda3/envs/polling/bin/python extract_britain_elects.py;
printf "executing analyse_polling_data.ipynb..\n";
/home/ubuntu/anaconda3/envs/polling/bin/jupyter nbconvert \
    --to notebook \
    --execute \
    /home/ubuntu/repo/polling-data-analysis/analyse_polling_data.ipynb && \
    printf "successfully executed analyse_polling_data.ipynb\n" && \
    mv /home/ubuntu/repo/polling-data-analysis/analyse_polling_data.nbconvert.ipynb \
    /home/ubuntu/repo/polling-data-analysis/analyse_polling_data.ipynb && \
    printf "analyse_polling_data.ipynb updated\n";
