#!/bin/bash

read -p "Enter network interface (default: ens33): " interface
interface=${interface:-ens33}

while true
do
    sudo python3 capture.py "$interface" 15
    sudo python3 predict.py
    sleep 30
done
