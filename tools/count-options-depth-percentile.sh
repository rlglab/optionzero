#!/bin/bash
which bc >/dev/null || { apt -qq update && apt -qq -y install bc || exit $?; } >/dev/null 2>&1
shopt -s lastpipe

sp_executable_file=$1; shift
tmp1=$(mktemp)
tmp2=$(mktemp)
tmp3=$(mktemp)
trap 'rm -f $tmp1 $tmp2 $tmp3' EXIT

grep -Eo "\]SP\[[0-9a-f]*\]" | tr -d "SP[]" | ${sp_executable_file} -mode decompress_str -conf_str program_quiet=true | tr ':' ' ' | sed -E "s/[0-9]+/+/g;s/-//g;s/^ //" | awk '{
  sum = 0
  max = 0
  for (i = 1; i <= NF; i++) {
    n = length($i)
    sum += n
    if (n > max) max = n
  }
  avg = sum / NF
  print avg, max
}' > $tmp1
cut -d' ' -f1 < $tmp1 | sort -n > $tmp2 # avg
cut -d' ' -f2 < $tmp1 | sort -n > $tmp3 # max

{
  for f in $tmp2 $tmp3; do # avg and max
    awk '{ sum += $1; count++ } END { print sum / count }' < $f
    ln=$(wc -l < $f)
    head -n1 $f # 0th percentile
    for idx in ${@:-"25 50 75 100"}; do
        ln_idx=$((idx*ln/100))
        awk -v ln_idx=$ln_idx 'NR == ln_idx { print $0 }' < $f
    done
  done
} | xargs
