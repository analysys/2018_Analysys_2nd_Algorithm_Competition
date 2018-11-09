#!/bin/bash

## 用来自动测试不同索引下的查询性能

# 879999987
# 879999987

primary_keys=(
    "uid,its"
    "uid,its,action_code"
)

indexs=(
    8192
    16384
    32768
    65536
)

default_pk="uid,its,action_code"
default_index_granularity=8192
ck_conn="localhost"

trap "kill -9 0; exit 0" INT HUP QUIT TERM

ck="clickhouse-client --host ${ck_conn}"
table="t_event"
dis_table="dis_event"
file_dir=$2 ##/data/clickhouse/yiguan/Analysys_olap_2018_demo.dat

function bench() {
    pk=$1
    index_granularity=$2

    echo "${table} pk : $pk ; index_granularity : ${index_granularity}"

    echo "DROP TABLE if exists ${table} on cluster logs" | $ck 
    echo "DROP TABLE if exists ${dis_table} on cluster logs;" | $ck

    echo "        
        CREATE TABLE ${table} on cluster logs 
        (
            uid String,
            its UInt32 CODEC(NONE),
            action_code  Enum8('evaluationGoods' = 1, 'browseGoods' = 2, 'viewOrder' = 3, 'startUp' = 4, 'unsubscribeGoods' = 5, 'login' = 6, 'order' = 7, 'reminding' = 8, 'addCart' = 9, 'consultGoods' = 10, 'searchGoods' = 11, 'shareGoods' = 12, 'orderPayment' = 13, 'collectionGoods' = 14, 'confirm' = 15),
            action_name String,
            event_city String,
            event_name String,
            event_brand String,
            event_price Float32 CODEC(NONE),
            event_nums Int8,
            event_how Int8,
            day Date
        ) 
        ENGINE = MergeTree PARTITION BY toYYYYMMDD(day) ORDER BY ($pk) SETTINGS index_granularity = ${index_granularity}" | $ck 

    ## metroHash64
    echo "CREATE TABLE ${dis_table}  on cluster logs as ${table}  ENGINE = Distributed(logs, default, t_event, metroHash64( toString(uid)) ); " | $ck

    if [ $? != 0 ];then
        exit 1
    fi 
   
    echo "start loading data"
	
    for file_path in `ls $file_dir | grep sort`;do
	echo "load  ${file_dir}/${file_path}" 
      ../importer -dsn=tcp://${ck_conn}:9000  -file=${file_dir}/${file_path} -logAll=false -table=${dis_table}  -pktype=String
    done


    if [ $? != 0 ];then
        exit 1
    fi 

    echo "start to sleep 100"
    sleep 100


    echo "start to optimize"
    echo "optimize table ${table} on cluster logs final" | $ck

    echo "start to benchmark"
    bash bench.sh ${table} ${ck_conn}
}

#! /bin/sh -
 
shell_name=`basename $0 .sh`

case $1 in 
    pk)
        echo "testing pk"
        for pk in ${primary_keys[@]};do
            bench "$pk" "$default_index_granularity"
        done
        ;;
    
    index)
        echo "testing indexs"
        for idx in ${indexs[@]};do
             bench "$default_pk" "$idx"
        done
        ;;
    
    none)
        echo "just testing"
        bench "$default_pk" "$default_index_granularity"
        ;;

    *)
        echo "Usage: $shell_name [pk|index]"
        exit 1
        ;;
esac