#!/bin/bash

# Author: Sören S. Dittrich
# Date: 04.06.2020
# Version: 0.0.1
# Description: Downloading caltech data 

seq=/usr/bin/seq
wget=/usr/bin/wget
tar=/bin/tar
rm=/usr/bin/rm

PATH="."
DL="http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/"
FILES="set"

if [ $# -eq 1 ]; then
    echo "Usage: $0 start end"
    exit -1
fi

# Download
for i in $($seq $1 $2); do
    if [ $i -lt 10 ]; then
        $wget "$DL$FILES"0"$i.tar"
    else
        $wget "$DL$FILES$i.tar"
    fi
done

# Untar
for i in $($seq $1 $2); do
    if [ $i -lt 10 ]; then
        $tar -xvf "$FILES"0"$i.tar"
    else
        $tar "-xvf  $FILES$i.tar"
    fi
done

# Delete
$rm *.tar
