#!/bin/bash
dir_path=$1
cd $dir_path

for i in *.jpg; do
	first=${i#IMG_}

	case "$first" in
		*_1.jpg) second=${first%_1.jpg}.jpg ;;
		*) second="$first" ;;
	esac
	echo original_name:$i
	echo changed_name:$second

	mv $i $(pwd)/"$second"

done
