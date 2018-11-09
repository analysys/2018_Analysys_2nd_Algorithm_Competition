#!/usr/bin/env bash

QUERIES_FILE="bench.sql"
TABLE=$1
CONN=$2
TRIES=3

cat "$QUERIES_FILE" | sed "s/{table}/${TABLE}/g" | while read query; do
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

    echo -n "["
    for i in $(seq 1 $TRIES); do
        RES=$(clickhouse-client --host "$CONN" --time --format=Null --query="$query" 2>&1)
        [[ "$?" == "0" ]] && echo -n "${RES}" || echo -n "null"
        [[ "$i" != $TRIES ]] && echo -n ", "
    done
    echo "],"
done
