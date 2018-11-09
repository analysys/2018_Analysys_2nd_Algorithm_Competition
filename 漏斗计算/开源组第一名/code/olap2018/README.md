## 2018 第二届易观OLAP算法大赛

-----------------------------

ClickHouse 设计 xFunnel 函数实现[复杂漏斗分析](http://ds.analysys.cn/ldjs.html)


### 数据预处理

* 网盘下载[数据集](http://ds.analysys.cn/ldjs.html)

* 编译数据处理Golang脚本
```
    go build -o importer bin/importer.go
```

### 编译ck

- 获取ClickHouse源码，修改代码后，
```
    ## patch 是从 b8543bcd4d0e9984405c75dbf00b23a6be727bc6 commit中切的分支修改
    git apply funnel.patch  

    ## 编译ck
    mkdir build
    cd build

    je=1
    tc=0
    BT=Release

    cmake  .. -DENABLE_VECTORCLASS=0   -DCMAKE_BUILD_TYPE=${BT} -DENABLE_POCO_ODBC=0  -DENABLE_JEMALLOC=${je}  -DENABLE_TCMALLOC=${tc}   -DUSE_INTERNAL_ZLIB_LIBRARY=0 -DUSE_INTERNAL_RDKAFKA_LIBRARY=0   -DCMAKE_CXX_COMPILER=`which g++-8` -DCMAKE_C_COMPILER=`which gcc-8`
    ninjar
```

- 配置ClickHouse集群，见ClickHouse官方文档


### 以下操作详细参见[workflow脚本](tools/workflow_batch.sh)

### 建单表
```
CREATE TABLE t_event on cluster logs 
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
        ENGINE = MergeTree PARTITION BY toYYYYMMDD(day) ORDER BY (uid,its,action_code) SETTINGS index_granularity = 8192"
```

### 建分布式表
```
CREATE TABLE dis_event on cluster logs as t_event  ENGINE = Distributed(logs, default, t_event, metroHash64( toString(uid)) );
```


### 导入数据
```
./importer -dsn=tcp://localhost:9000  -file=/data/file.log -logAll=false -table=dis_event  -pktype=String
```

### 查询

* 正式比赛六题，详见[prod](prod) 目录
* 示例查询
1、计算出20180601-20180610范围内，依次有序触发“addCart-加入购物车”、“order-生成订单”、“orderPayment-订单付款”、“evaluationGoods-评价商品”的用户转换情况以及各步骤转换时间中位数，且满足时间窗口为1天，且要求“order-生成订单”与“evaluationGoods-评价商品”对应的“brand-商品品牌”属性相同。


```
SELECT
    day,
    countIf(level >= 1) AS _1,
    countIf(level >= 2) AS _2,
    countIf(level >= 3) AS _3,
    countIf(level >= 4) AS _4,
    medianExactIf( toFloat32(x[2].1 - x[1].1), level >= 2) AS median1,
    medianExactIf( toFloat32(x[3].1 - x[2].1), level >= 3) AS median2,
    medianExactIf( toFloat32(x[4].1 - x[3].1), level >= 4) AS median3
FROM
(
    SELECT
        x[1].2 AS day,
        length(x) AS level,
        x
    FROM
    (
        SELECT arrayJoin(xFunnel(86400, 2, '2.3=4.3')((its, day, event_brand), action_code = 'addCart', action_code = 'order', action_code = 'orderPayment', action_code = 'evaluationGoods')) AS x
        FROM dis_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-06-10') AND (action_code IN ('addCart', 'order', 'orderPayment', 'evaluationGoods'))
        GROUP BY uid
    )
)
GROUP BY day order by day

```

在正式比赛中，上面查询耗时 0.6s