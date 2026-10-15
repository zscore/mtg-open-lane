#!/bin/zsh
# Pure-WheelNet baseline on all 19 sets, one features-only multi-set WheelNet per pack size, 20k training drafts
# per set, 3 epochs; queued after TableNet run B (nohup caffeinate -i zsh results/tablenet/run_wheelnet_allsets.sh &)
source .venv/bin/activate
while pgrep -f "name=tablenet_whist_hist_w3" > /dev/null; do sleep 60; done
python scripts/train_wheelnet_allsets.py DFT,DSK,FDN,FIN,FRA,HOB,MSH,OTJ,SOS,TDM,TLA,TMT,WOE 3 20000 \
    > results/tablenet/wheelnet_allsets_14.log 2>&1
python scripts/train_wheelnet_allsets.py BLB,ECL,EOE,MKM 3 20000 > results/tablenet/wheelnet_allsets_13.log 2>&1
python scripts/train_wheelnet_allsets.py LCI,MOM 3 20000 > results/tablenet/wheelnet_allsets_15.log 2>&1
