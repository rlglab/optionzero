#!/bin/bash
which bc >/dev/null || { apt -qq update && apt -qq -y install bc || exit $?; } >/dev/null 2>&1
shopt -s lastpipe

total=0
repeat=0
nonrep=0
grep -E "^\+.+-.+" | while read -r type option count; do
    if (( $(printf "%s\n" ${option//-/ } | sort -u | wc -l) > 1 )); then
        nonrep=$((nonrep+count))
    else
        repeat=$((repeat+count))
    fi
    total=$((total+count))
done
{
echo \#=$total
echo \#repeat=$repeat
echo \$nonrep=$nonrep
echo %repeat=$(bc -l <<< $repeat/$total)
echo %nonrep=$(bc -l <<< $nonrep/$total)
} | stdbuf -o0 sed -E "s/=\./=0./g;s/[+-]?nan/0/" | while IFS= read -r stat; do
	item=${stat%=*}
	data=${stat#*=}
	[[ $data ]] && case $item in
		μ*|±*) data=$(printf "%.0f" $data); ;;
		%*) data=$(printf "%.2f%%" $(<<< $data*100 bc -l)); ;;
	esac
	[[ $item ]] && echo $item=$data || echo
done 