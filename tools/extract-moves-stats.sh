#!/bin/bash
# LEN=3

which bc >/dev/null || { apt -qq update && apt -qq -y install bc || exit $?; } >/dev/null 2>&1
shopt -s lastpipe
sum() { xargs | tr ' ' '+' | bc -l; }
average() { file=$(mktemp); cat $1 | xargs | tr '\n' ' ' | sed -E "s/ +/+/g" > $file; bc -l <<< "($(cat $file)0)/$(grep -o + $file | wc -l)" | sed -E "s/\.?0+$//"; rm $file; }
stddev() { [[ ${1^^} == *P ]] && n="(n)" || n="(n-1)"; xargs | tr ' ' '\n' | awk '{x[NR]=$0; s+=$0; n++} END{a=s/n; for (i in x){ss += (x[i]-a)^2} sd = sqrt(ss/'"$n"'); print sd}'; }
output() { cat; }
{
tag=${1:-+}
declare -A A Ap Ao L
if (( LEN )); then
	for op in $(seq 1 $LEN); do
		L[$op]=""
	done
fi
grep "^$tag" | sed -E "s/\\$tag +//" | while read -r move count; do
	A[$move]=$count
	[[ $move == *-* ]] && Ao[$move]=$count || Ap[$move]=$count
	numop=(${move//-/ })
	L[${#numop[@]}]+=$move:$count$'\n'
done

N=$(($(echo ${A[@]} | tr ' ' '+')))
Np=$(($(echo ${Ap[@]} | tr ' ' '+')))
No=$(($(echo ${Ao[@]} | tr ' ' '+')))
echo \#=$N
echo \#p=$Np
echo \#o=$No
for l in $(printf "%s\n" ${!L[@]} | sort -n); do
	echo \#o[$l]=$(<<< ${L[$l]:-0} cut -d: -f2 | sum)
done
echo %p=$(bc -l <<< "$Np/$N")
echo %o=$(bc -l <<< "$No/$N")
for l in $(printf "%s\n" ${!L[@]} | sort -n); do
	echo %o[$l]=$(bc -l <<< "$(<<< ${L[$l]:-0} cut -d: -f2 | sum)/$N" 2>/dev/null)
done
echo

count_num_th() {
	[[ $2 ]] || return
	for n in ${@:2}; do
		(( $(<<< "$n $1" bc -l) )) && echo $n
	done | wc -l
}
ths=(0.5 0.75 1 2 5 10)
fetch_top_k() {
	printf "%s\n" ${@:2} | sort -n | tail -n$1
}
ks=(1 2 3 5 10 20 50 100 200)
kps=(1 5 10 25 50 75 90 95 99)

up=($(<<< ${Ap[@]} average) $(<<< ${Ap[@]} stddev))
bp=()
for i in {0..3}; do
	bp[$i]=$(printf "%s\n" "${up[0]}-$i*${up[1]}" "${up[0]}+$i*${up[1]}" | bc -l | xargs | tr ' ' ':')
done
echo μ\#p=${up[0]}
echo ±\#p=${up[1]}
echo \#{p}=${#Ap[@]}
for i in {1..3}; do
	echo \#{p±${i}}=$(for a in ${Ap[@]}; do printf -- "${bp[$i]/:/<=%s && %s<=}\n" $a $a | bc; done | sum)
done
for th in ${ths[@]}; do
	echo \#{p≥${th}μ}=$(count_num_th ">= $(<<< "$th * $up" bc -l)" ${Ap[@]})
done
for k in ${ks[@]}; do
	echo %{p-$k}=$(<<< "$(fetch_top_k $k ${Ap[@]} | sum)/$Np" bc -l)
done
for kp in ${kps[@]}; do
	k=$(<<< "${#Ap[@]} * $kp / 100" bc -l | sed -E "s/^\./0./;s/\.?0+$//;s/^$/1/")
	#(( $(<<< "$k < 1" bc -l) )) && k=1 || k=${k%.*}
	(( $(<<< "$k - ${k%.*} > 0" bc -l) )) && k=$((${k%.*}+1)) || k=${k%.*}
	echo %{p-$kp%}=$(<<< "$(fetch_top_k $k ${Ap[@]} | sum)/$Np" bc -l)
done
echo

uo=($(<<< ${Ao[@]} average 2>/dev/null) $(<<< ${Ao[@]} stddev 2>/dev/null))
bo=()
(( ${#Ao[@]} )) && for i in {0..3}; do
	bo[$i]=$(printf "%s\n" "${uo[0]}-$i*${uo[1]}" "${uo[0]}+$i*${uo[1]}" | bc -l | xargs | tr ' ' ':')
done || uo=()
echo μ\#o=${uo[0]}
echo ±\#o=${uo[1]}
echo \#{o}=${#Ao[@]} | sed "s/=0/=/"
for i in {1..3}; do
	echo \#{o±${i}}=$(for a in ${Ao[@]}; do printf -- "${bo[$i]/:/<=%s && %s<=}\n" $a $a | bc; done | sum)
done
for th in ${ths[@]}; do
	echo \#{o≥${th}μ}=$(count_num_th ">= $(<<< "$th * $uo" bc -l 2>/dev/null)" ${Ao[@]})
done
for k in ${ks[@]}; do
	echo %{o-$k}=$(<<< "$(fetch_top_k $k ${Ao[@]} | sum)/$No" bc -l 2>/dev/null)
done
for kp in ${kps[@]}; do
	k=$(<<< "${#Ao[@]} * $kp / 100" bc -l | sed -E "s/^\./0./;s/\.?0+$//;s/^$/1/")
	#(( $(<<< "$k < 1" bc -l) )) && k=1 || k=${k%.*}
	(( $(<<< "$k - ${k%.*} > 0" bc -l) )) && k=$((${k%.*}+1)) || k=${k%.*}
	echo %{o-$kp%}=$(<<< "$(fetch_top_k $k ${Ao[@]} | sum)/$No" bc -l 2>/dev/null)
done

for l in $(printf "%s\n" ${!L[@]} | sort -n); do
	unset Ao
	declare -A Ao
	while read -r move count; do
		[[ $move ]] && Ao[$move]=$count
	done <<< ${L[$l]//:/ }
	No=$(($(echo ${Ao[@]} | tr ' ' '+')))
	uo=($(<<< ${Ao[@]} average 2>/dev/null) $(<<< ${Ao[@]} stddev 2>/dev/null))
	bo=()
	(( ${#Ao[@]} )) && for i in {0..3}; do
		bo[$i]=$(printf "%s\n" "${uo[0]}-$i*${uo[1]}" "${uo[0]}+$i*${uo[1]}" | bc -l | xargs | tr ' ' ':')
	done || uo=()
	echo
	echo μ\#o[$l]=${uo[0]}
	echo ±\#o[$l]=${uo[1]}
	echo \#{o[$l]}=${#Ao[@]} | sed "s/=0/=/"
	for i in {1..3}; do
		echo \#{o[$l]±${i}}=$(for a in ${Ao[@]}; do printf -- "${bo[$i]/:/<=%s && %s<=}\n" $a $a | bc; done | sum)
	done
	for th in ${ths[@]}; do
		echo \#{o[$l]≥${th}μ}=$(count_num_th ">= $(<<< "$th * $uo" bc -l 2>/dev/null)" ${Ao[@]})
	done
	for k in ${ks[@]}; do
		echo %{o[$l]-$k}=$(<<< "$(fetch_top_k $k ${Ao[@]} | sum)/$No" bc -l 2>/dev/null)
	done
	for kp in ${kps[@]}; do
		k=$(<<< "${#Ao[@]} * $kp / 100" bc -l | sed -E "s/^\./0./;s/\.?0+$//;s/^$/1/")
		#(( $(<<< "$k < 1" bc -l) )) && k=1 || k=${k%.*}
		(( $(<<< "$k - ${k%.*} > 0" bc -l) )) && k=$((${k%.*}+1)) || k=${k%.*}
		echo %{o[$l]-$kp%}=$(<<< "$(fetch_top_k $k ${Ao[@]} | sum)/$No" bc -l 2>/dev/null)
	done
done
} | stdbuf -o0 sed -E "s/=\./=0./g;s/[+-]?nan/0/" | while IFS= read -r stat; do
	item=${stat%=*}
	data=${stat#*=}
	[[ $data ]] && case $item in
		μ*|±*) data=$(printf "%.0f" $data); ;;
		%*) data=$(printf "%.2f%%" $(<<< $data*100 bc -l)); ;;
	esac
	[[ $item ]] && echo $item=$data || echo
done | output
