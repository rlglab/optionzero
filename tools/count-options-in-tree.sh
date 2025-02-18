#!/bin/bash

which bc >/dev/null || { apt -qq update && apt -qq -y install bc || exit $?; } >/dev/null 2>&1
shopt -s lastpipe

sp_executable_file=$1
Lsum=()
Lmax=()
N=()
N_opt=()
N_opt_sim=()
N_type_opt=()
while IFS= read -r sgf; do
    sgf=${sgf:1:-1}
    [[ $sgf =~ DLEN\[[0-9]+-([0-9]+)\] ]] || exit $?
    len=$((BASH_REMATCH[1]+1))
    echo $((++ID)): $len moves >&2

    moves=$(tr ';' '\n' <<< $sgf | grep "^B")
    moves_nl_SP=$(grep -oEn "\]SP\[[0-9a-f]*\]" <<< $moves)
    moves_nl_C=$(grep -oEn "\]C\[[0-9]*\]" <<< $moves | tr -d "C[]")
    moves_nl_OP=$(grep -oEn "\]OP\[[0-9:-]*\]" <<< $moves | tr -d "OP[]")
    [[ $(wc -l <<< $moves_nl_SP ) == $(wc -l <<< $moves_nl_C ) ]] || { echo mismatch SP/C >&2; exit 100; }
    [[ $(cut -d: -f1 <<< $moves_nl_SP | tr '\n' -) == $(cut -d: -f1 <<< $moves_nl_C | tr '\n' - ) ]] || { echo mismatch C/OP >&2; exit 100; }
    [[ $(wc -l <<< $moves_nl_C ) == $(wc -l <<< $moves_nl_OP ) ]] || { echo mismatch C/OP >&2; exit 100; }
    [[ $(cut -d: -f1 <<< $moves_nl_C | tr '\n' -) == $(cut -d: -f1 <<< $moves_nl_OP | tr '\n' - ) ]] || { echo mismatch C/OP >&2; exit 100; }

    paste \
        <(cut -d: -f1 <<< $moves_nl_SP | sed "s/$/-1/" | bc) \
        <(cut -d: -f2 <<< $moves_nl_SP | tr -d "SP[]" | ${sp_executable_file} -mode decompress_str -conf_str program_quiet=true | tr ':' ' ' | sed -E "s/[0-9]+/+/g;s/-//g" | awk '{
        max = 0;
        for (i = 1; i <= NF; i++) { if (length($i) > max) { max = length($i); } }
        print max;
    }') | while read -r idx l; do
            Lsum[$idx]=$((Lsum[$idx]+$l))
            (( l > Lmax[$idx] )) && Lmax[$idx]=$l
            N[$idx]=$((N[$idx]+1))
        done

    paste \
        <(cut -d: -f1 <<< $moves_nl_C | sed "s/$/-1/" | bc) \
        <(cut -d: -f2 <<< $moves_nl_C) \
        <(cut -d: -f2- <<< $moves_nl_OP | awk -F: '{ delete u; for (i = 1; i <= NF; i++) { u[$i] = 1; }; print length(u); }') | \
        while read -r idx OP_count uniq_OP_num; do
            N_opt[$idx]=$((N_opt[$idx]+(OP_count>0?1:0)))
            N_opt_sim[$idx]=$((N_opt_sim[$idx]+OP_count))
            N_type_opt[$idx]=$((N_type_opt[$idx]+uniq_OP_num))
        done
done

[[ ${#Lsum[@]} == ${#Lmax[@]} && ${#Lmax[@]} == ${#N[@]} ]] || exit 101

N_total=$(bc <<< $(printf "%s+" ${N[@]})0)
Lsum_total=$(bc <<< $(printf "%s+" ${Lsum[@]})0)
Lavg=($(paste <(printf "%s\n" ${Lsum[@]}) <(printf "%s\n" ${N[@]}) | tr '\t' '/' | bc -l))
Lmax_max=$(printf "%s\n" ${Lmax[@]} | sort -n | tail -n1)
N_opt_total=$(bc <<< $(printf "%s+" ${N_opt[@]})0 2>/dev/null)
if (( N_opt_total )); then
    N_opt_sim_total=$(bc <<< $(printf "%s+" ${N_opt_sim[@]})0 2>/dev/null)
    N_type_opt_total=$(bc <<< $(printf "%s+" ${N_type_opt[@]})0 2>/dev/null)
    type_opt_avg=($(paste <(printf "%s\n" ${N_type_opt[@]}) <(printf "%s\n" ${N_opt[@]}) | tr '\t' '/' | sed -E "s/.+\/0$/0/" | bc -l 2>/dev/null))
    paste \
        <(printf "%s\n" idx "*" $(seq 0 $((${#N[@]}-1)))) \
        <(printf "%s\n" num $N_total ${N[@]}) \
        <(printf "%s\n" avg $(bc -l <<< $Lsum_total/$N_total) ${Lavg[@]}) \
        <(printf "%s\n" max $Lmax_max ${Lmax[@]}) \
        <(printf "%s\n" num_opt $N_opt_total ${N_opt[@]}) \
        <(printf "%s\n" num_opt_sim $N_opt_sim_total ${N_opt_sim[@]}) \
        <(printf "%s\n" avg_type_opt $(bc -l <<< $N_type_opt_total/$N_opt_total) ${type_opt_avg[@]})
else
    paste \
        <(printf "%s\n" idx "*" $(seq 0 $((${#N[@]}-1)))) \
        <(printf "%s\n" num $N_total ${N[@]}) \
        <(printf "%s\n" avg $(bc -l <<< $Lsum_total/$N_total) ${Lavg[@]}) \
        <(printf "%s\n" max $Lmax_max ${Lmax[@]})
fi