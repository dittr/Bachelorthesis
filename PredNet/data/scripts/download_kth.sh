#!/bin/bash

# Author: Sören S. Dittrich
# Date: 02.06.2020
# Version: 0.0.1
# Description: Downloading kth data

wget=/usr/bin/wget
unzip=/usr/bin/unzip
rm=/usr/bin/rm

PATH="."
DL="http://www.nada.kth.se/cvap/actions/"
ACTIONS=("boxing" "handclapping" "handwaving" "jogging" "running" "walking")

# Use all actions if no parameter is given through command line
if [ $# -eq 0 ]; then
    actions=${ACTIONS[*]}
# Use only the given parameters
else
    for action in "$@"; do
        for given in ${ACTIONS[*]}; do
            if [ $action == $given ]; then
                actions+=" $given"
            fi
        done
    done
fi

# Download
for action in ${actions[*]}; do
    $wget "$DL$action.zip"
done
    
# Unzip
for action in ${actions[*]}; do
    $unzip "$action.zip" -d $action
done

# Remove zip files
$rm *.zip
