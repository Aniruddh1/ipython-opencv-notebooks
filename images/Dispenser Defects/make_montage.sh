#!/bin/bash -eu

base="./"
find "$base" -type d | while read fld
do
  [ $fld == $base ] && continue
  echo ">>> $fld"
  aux="${fld//\//_}"
  aux="${aux:1}"
  find "$fld" -type f -maxdepth 1 -iregex ".*\.\(PNG\|png\)" > "/tmp/${aux}.txt"

  if [ -s "/tmp/${aux}.txt" ]
  then
    montage "@/tmp/${aux}.txt" -geometry 64x -tile 3x "${aux}.jpg"
  fi
  rm "/tmp/${aux}.txt"
done