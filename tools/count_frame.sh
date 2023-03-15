#!/bin/bash

directory_path="/home/kanya/Data/MMPdata/validation/labels"
declare -a arr_f=("64pm")

for f in "${arr_f[@]}"
  do
  for i in {0..8}
  do
    folder=${directory_path}/${f}/retail_${i}
    file_count=$(ls -1 $folder | wc -l)
    echo "Folder: $folder"
    echo "File count: $file_count"
  done
done
