input_dir=$1
output_dir_1=$2
output_dir_2=$3
output_dir=$4

for i in $input_dir/*.jpg; do
	file_name=${i#*e/}
	file_name=${file_name%.jpg}
	echo $file_name
	output_dir_1_file=$output_dir_1/"$file_name"_output.png
	output_dir_2_file=$output_dir_2/"$file_name"_anno.jpg
	echo output_dir_1_file: $output_dir_1_file
	echo output_dir_2_file: $output_dir_2_file
	output_file=$4/"$file_name"_allresult.png
	echo output_file:$output_file
	convert +append $output_dir_1_file $output_dir_2_file $output_file

done
