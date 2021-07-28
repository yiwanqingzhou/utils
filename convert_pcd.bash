#!/bin/bash

function help()
{
   # Display Help
    echo "Convert ASCII PCD files in directory to binary"
    echo
    echo "usage: convert_pcd.bash <directory> [<keep>]"
    echo "where <keep> is optional (default is 0)"
    echo "   0: don't keep original"
    echo "   1: keep original (stored as <filename>.ascii)"
    echo
}

DIR=$1
shift 1
KEEP=$1
shift 1

if ! command -v pcl_convert_pcd_ascii_binary &> /dev/null
then
    echo "pcl_convert_pcd_ascii_binary could not be found, pcl-tools package is required to be installed"
    exit
fi

if [ -z "$KEEP" ]; then
    KEEP="0"
fi

if [ ! -d "$DIR" ]; then
    echo "Invalid directory provided: $DIR"
    help
    exit
fi

for f in ${DIR}/*.pcd; do
    if [ ! -f "$f" ]; then
        continue
    fi
    BINARY=`file -b --mime-encoding $f`
    if [ "$BINARY" == "binary" ]; then
        echo "Already binary: $f"
        continue
    fi
    if [ "${KEEP}" == "1" ]; then
        ASCII=${f}.ascii
        echo "moving $f to $ASCII"
        mv $f $ASCII
    elif [ "${KEEP}" == "0" ]; then
        ASCII=$f
    else
        echo "Please specify <keep> argument as either 0 (don't keep) or 1 (keep)"
        exit
    fi
    echo "${ASCII}"
    echo "${f}"
    pcl_convert_pcd_ascii_binary $ASCII $f 1
done
