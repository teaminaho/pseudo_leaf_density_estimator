#!/bin/bash -e
# @(#) This script is main.py run for any datas.;if you want to run any data, you put atribute as directory of data you want to process.

input_dir=${1}
output_dir=$(dirname $0)/data/output
roi_csv=${1}/roi.csv
echo $output_dir

# roi.csv is exist
if [ -e $roi_csv ]; then
	cat ${roi_csv} | while IFS=, read file_name highest lowest; do
		if test ${file_name:0:1} = "#"; then
			echo $file_name
			dir_name=${file_name:1}
			echo $dir_name
			if [ -e $output_dir/$dir_name ]; then
				echo "already existed dir"
			else
				mkdir $output_dir/$dir_name
			fi
		else
			./main.py "${input_dir}/$dir_name/${file_name}" --hmin ${highest} --hmax ${lowest}
			mv "${output_dir}/${file_name%%.jpg}_output.png" "${output_dir}/$dir_name"
		fi
	done
# roi.csv is not exist
else
	for img_path in ${input_dir}/*.jpg; do
		img_name=${img_path#${input_dir}/}
		./main.py $img_path --hmin 0 --hmax 3264
		output_path=$output_dir/${img_path#data/}
		echo output_path:$output_path
		mv "${output_dir}/${img_name%%.jpg}_output.png" "${output_dir}/$img_name"
	done
fi
