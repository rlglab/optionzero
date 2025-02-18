#!/bin/bash
set -e

usage()
{
	echo "Usage: $0 GAME_TYPE CONFIGURE_FILE END_ITERATION [OPTION]..."
	echo "The zero-server manages the training session and organizes connected zero-workers."
	echo ""
	echo "Required arguments:"
	echo "  GAME_TYPE: $(find ./ ../ -maxdepth 2 -name build.sh -exec grep -m1 support_games {} \; -quit | sed -E 's/.+\("|"\).*//g;s/" "/, /g')"
	echo "  CONFIGURE_FILE: the configure file (*.cfg) to use"
	echo "  END_ITERATION: the total number of iterations for training"
	echo ""
	echo "Optional arguments:"
	echo "  -h,        --help                 Give this help list"
	echo "  -n,        --name                 Assign name for training directory"
	echo "  -np,       --name_prefix          Add prefix name for default training directory name"
	echo "  -ns,       --name_suffix          Add suffix name for default training directory name"
	echo "  -g,        --gpu                  Assign the GPU for network model initialization, e.g. 0"
	echo "             --sp_executable_file   Assign the path for self-play executable file"
	echo "             --op_executable_file   Assign the path for optimization executable file"
	echo "             --link_sgf             Assign the path of sgf for training without self play (only op)"
	echo "  -conf_str                         Overwrite settings in the configure file"
	exit 1
}

# check argument
if [ $# -lt 3 ] || [ $(($# % 2)) -eq 0 ]; then
	usage
else
	game_type=$1; shift
	configure_file=$1; shift
	end_iteration=$1; shift
fi

train_dir=""
name_prefix=""
name_suffix=""
gpu_list=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | sort -k2n -k3n | head -1 | cut -d, -f1)
sp_executable_file=build/${game_type}/minizero_${game_type}
op_executable_file=minizero/learner/train.py
overwrite_conf_str=""
link_sgf=""
while :; do
	case $1 in
		-h|--help) shift; usage
		;;
		-n|--name) shift; train_dir=$1
		;;
		-np|--name_prefix) shift; name_prefix=$1
		;;
		-ns|--name_suffix) shift; name_suffix=$1
		;;
		-g|--gpu) shift; gpu_list=$1
		;;
		--sp_executable_file) shift; sp_executable_file=$1
		;;
		--op_executable_file) shift; op_executable_file=$1
		;;
		--link_sgf) shift; link_sgf=$1
		;;
		-conf_str) shift; overwrite_conf_str=$1
		;;
		"") break
		;;
		*) echo "Unknown argument: $1"; usage
		;;
	esac
	shift
done

# create default name; also check if configurations are valid
testrun_stderr_tmp=$(mktemp)
default_name=$(${sp_executable_file} -mode zero_training_name -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" 2>${testrun_stderr_tmp} || :)
testrun_stderr=$(<${testrun_stderr_tmp})
rm -f ${testrun_stderr_tmp}
if [[ ! ${default_name} ]]; then
	echo "${testrun_stderr}" >&2
	exit 1
fi
# use default name of training if name is not assigned
if [[ -z ${train_dir} ]]; then
	train_dir=${name_prefix}${default_name}${name_suffix}
fi

run_stage="R"
if [ -d ${train_dir} ]; then
	read -n1 -p "${train_dir} has existed. (R)estart / (C)ontinue / (Q)uit? " run_stage
	echo ""
fi

zero_start_iteration=1
model_file="weight_iter_0.pt"
if [[ ${run_stage,} == "r" ]]; then
	rm -rf ${train_dir}

	echo "create ${train_dir} ..."
	mkdir -p ${train_dir}/model ${train_dir}/sgf
	if [[ ! -z ${link_sgf} ]];
	then
		ln ${link_sgf}/* ${train_dir}/sgf/
		end_iteration=$(ls ${train_dir}/sgf/ | wc -l)
		echo "link ${link_sgf} ..."
		echo "end_iteration: ${end_iteration}"
	fi
	touch ${train_dir}/op.log
	new_configure_file=$(basename ${train_dir}).cfg
	${sp_executable_file} -gen ${train_dir}/${new_configure_file} -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" 2>/dev/null

	# setup initial weight
	cuda_devices=$(echo ${gpu_list} | awk '{ split($0, chars, ""); printf(chars[1]); for(i=2; i<=length(chars); ++i) { printf(","chars[i]); } }')
	echo "train \"\" -1 -1" | CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python ${op_executable_file} ${game_type} ${train_dir} ${train_dir}/${new_configure_file} >/dev/null 2>&1
elif [[ ${run_stage,} == "c" ]]; then
	zero_start_iteration=$(ls ${train_dir}/model/ | grep ".pt$" | wc -l)
	model_file=$(ls ${train_dir}/model/ | grep ".pt$" | sort -V | tail -n1)
	new_configure_file=$(basename ${train_dir}/*.cfg)
	echo y | ${sp_executable_file} -gen ${train_dir}/${new_configure_file} -conf_file ${train_dir}/${new_configure_file} -conf_str "${overwrite_conf_str}" 2>/dev/null

	# friendly notification if continuing training
	read -n1 -p "Continue training from iteration: ${zero_start_iteration}, model file: ${model_file}, configuration: ${train_dir}/${new_configure_file}. Sure? (y/n) " yn
	[[ ${yn,,} == "y" ]] || exit
	echo ""
else
	exit
fi

# run zero server
conf_str="zero_training_directory=${train_dir}:zero_end_iteration=${end_iteration}:nn_file_name=${model_file}:zero_start_iteration=${zero_start_iteration}"
if [[ ! -z ${link_sgf} ]];
then
	conf_str="$conf_str:zero_num_games_per_iteration=0"
fi
${sp_executable_file} -conf_file ${train_dir}/${new_configure_file} -conf_str ${conf_str} -mode zero_server

# option analysis
op_length=$(cat ${train_dir}/${new_configure_file} | grep -oE "option_seq_length=[0-9]*" | grep -oE "[0-9]*")
if [[ $op_length -gt 1 ]]; then
	op_dir=${train_dir}/option_analysis
	op_sgf_dir=${op_dir}/latest_100_games_sgf
	stats_dir=${op_dir}/stats
	mkdir -p $stats_dir

	tools/fetch-complete-latest.sh $train_dir 100 ${op_sgf_dir}
	cat ${op_sgf_dir}/*.sgf | tools/count-moves.sh > ${stats_dir}/moves.txt
	cat ${stats_dir}/moves.txt | LEN=$op_length tools/extract-moves-stats.sh > ${stats_dir}/moves-stats.txt
	cat ${stats_dir}/moves.txt | tools/extract-repeated-options.sh > ${stats_dir}/repeated-options.txt
	cat ${op_sgf_dir}/*.sgf | tools/count-options-in-tree.sh ${sp_executable_file} > ${stats_dir}/options-in-tree.txt
	cat ${op_sgf_dir}/*.sgf | tools/count-options-depth-percentile.sh ${sp_executable_file} 50 100 > ${stats_dir}/options-depth-percentile.txt
	python tools/option_analysis.py -in_dir $train_dir
fi
