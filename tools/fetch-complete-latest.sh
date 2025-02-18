#!/bin/bash
repo=${1?}
limit=${2:-100}
output_dir=${3}
num_jobs=${4:-10}

[ -d $repo/sgf ] || exit $?
[[ $output_dir ]] && mkdir -p $output_dir
shopt -s lastpipe
tmp=$(mktemp -d -t fetch-complete-latest.XXXXXX)
trap "cleanup" EXIT
cleanup() {
    jobs -r -p | xargs -r kill
    rm -rf $tmp
}
sgf() {
    if [ ! -s $tmp/$1.sgf ]; then
        if [ ! -e $repo/sgf/$1.sgf ]; then
            echo /dev/null
            return 1
        fi
        flock -n $tmp/$1.sgf.tmp -c "sed 's/OBS\[[0-9a-f]*\]/OBS[]/g' $repo/sgf/$1.sgf > $tmp/$1.sgf.tmp && mv $tmp/$1.sgf.tmp $tmp/$1.sgf"
    fi
    while [ ! -s $tmp/$1.sgf ]; do sleep 0.1; done
    echo $tmp/$1.sgf
    return 0
}
sign() {
    grep -Eo ";[BW]\[[0-9]+\]" | xargs | tr -d ' '
}
extract_part() {
    local part=$1
    local start=$2
    local end=$3
    #part=$(<<< $part sed -E "s/$(printf "%.0s;[^;]+" $(seq $((start+1))))//" | grep -Eo -m1 "$(printf "%.0s;[^;]+" $(seq $((end-start+1))))")
    part=${part:$(grep -o . <<< $part | grep -n -m$((start+1+1)) ";" | tail -n1 | cut -d: -f1)-1}
    part=${part:0:$(grep -o . <<< "$part;" | grep -n -m$((end-start+1+1)) ";" | tail -n1 | cut -d: -f1)-1}
    echo "$part"
}
log() {
    local nf=
    [[ $1 == -n ]] && shift && nf=-n
    log+="$@"
    [[ $nf && ${num_jobs:-1} -gt 1 ]] && return
    echo $nf "$log" >&2
    log=
}

latest_iter=${latest_iter:-$(ls -t $repo/sgf | sort -rn | head -n1 | sed 's/\.sgf//')}
for iter in $(seq $latest_iter -1 1); do
    (( count >= limit )) && break
    sgf $iter >/dev/null || continue
    nl $(sgf $iter) | tac | grep "#$" | while IFS= read -r res && (( count < limit )); do
        count=$((count+1))
        ID=$iter-$(<<< $res cut -f1 | xargs)
        complete=$(<<< $res cut -f2-)
        [[ $complete =~ DLEN\[([0-9]+)-([0-9]+)\] ]]
        start=${BASH_REMATCH[1]:-0}
        end=${BASH_REMATCH[2]:-0}
        header=$(grep -Eo "(GM|RE|SD|DLEN)\[[^]]+\]" <<< $complete | sed -E "s/DLEN\[[0-9]+-/DLEN[0-/" | xargs | tr -d ' ')
        if [[ $output_dir ]]; then
            output=$output_dir/$ID.sgf
            [ -e $output ] && log "#$ID $header: [-]" && continue
            touch $output
        fi
        ( # parallel execution block
            log -n "#$ID $header: ${start}-${end}"
            [[ $output ]] && trap "[ -s $output ] || rm -f $output" EXIT
            if (( start )); then
                [[ $complete =~ SD\[[0-9]+\] ]]
                SD=${BASH_REMATCH[0]}
                sign_complete=$(sign <<< $complete)
                complete_new=$(extract_part "$complete" $start $end)
                search_iter=$iter
                buffer=()
                while (( start )); do
                    end_target=$((start-1))
                    part=
                    while [[ ! $part ]]; do
                        while [[ ! $buffer ]] && sgf $search_iter >/dev/null; do
                            tac $(sgf $search_iter) | grep -v "#$" | grep -F "${SD:-;}" | while IFS= read -r buf; do
                                buffer+=("$buf")
                            done
                            search_iter=$((search_iter-1))
                        done
                        part=${buffer[0]:-$(sed -E "s/DLEN\[[^]]+\]/DLEN[0-${end_target}]/" <<< $complete)}
                        buffer=("${buffer[@]:1}")
                        [[ $part =~ DLEN\[([0-9]+)-([0-9]+)\] ]]
                        start=${BASH_REMATCH[1]:-0}
                        end=${BASH_REMATCH[2]:-0}
                        [[ $SD && $end == $end_target ]] && break
                        [[ "${sign_complete}" != "$(sign <<< $part)"* ]] && part= && continue
                        [[ $end == $end_target ]] && break
                        buffer=("$part" "${buffer[@]}")
                        start=$((end+1))
                        end=${end_target}
                        part=$(sed -E "s/DLEN\[[^]]+\]/DLEN[${start}-${end}]/" <<< $complete)
                    done
                    complete_new=$(extract_part "$part" $start $end)${complete_new}
                    log -n " ${start}-${end}"
                done
                complete_new=$(<<< $complete grep -Eo "\(;[^;]+" | sed -E "s/DLEN\[[0-9]+-/DLEN[0-/")${complete_new}
                complete=$complete_new
                if [[ "${sign_complete}" != "$(sign <<< $complete)" ]]; then
                    log -n " [x]"
                    complete=
                fi
            fi
            if (( $(<<< $complete tr ';' '\n' | grep "^[BW]" | grep -Fv "]P[" | wc -l) )); then
                log -n " [!]"
                complete=
            fi
            log
            [[ $complete ]] && echo "$complete" >> ${output:-/dev/stdout}
        ) &
        while (( $(jobs -r -p | wc -l) >= ${num_jobs:-1}+1 )); do wait -n; done
    done
done
wait
