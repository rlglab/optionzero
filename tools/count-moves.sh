#!/bin/bash
# ID=N:F:U:R:L:D:UR:UL:DR:DL:UF:RF:LF:DF:URF:ULF:DRF:DLF

shopt -s lastpipe
ID=($(tr ':' ' ' <<< ${ID:-N:F:U:R:L:D:UR:UL:DR:DL:UF:RF:LF:DF:URF:ULF:DRF:DLF}))
NUM=${#ID[@]}
moves=$(tr "(;)" '\n' | grep "^B" | sed "s/$/OP1[$NUM]/"; echo)
paste <(grep -Eo "^B\[[0-9]+\]" <<< $moves | tr -d "B[]") <(grep -Eo "\]OP1\[[0-9-]+\].*" <<< $moves | cut -b6- | cut -d']' -f1) | tr '\t' ' ' | sed "s/^$NUM //;s/ $NUM$//" | \
while IFS= read -r res; do
    res=($res)
    echo +${res[0]}
    echo :${res[-1]}
done | sort | uniq -c | while read -r count move; do
    echo $move $count
done | sort -V | while read -r move count; do
    type=${move:0:1}
    move=${move:1}
    echo $type $(for a in $(tr '-' ' ' <<< $move); do
        echo ${ID[$a]}
    done | xargs | tr ' ' '-') $count
done
