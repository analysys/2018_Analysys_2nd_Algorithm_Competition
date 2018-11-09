

-- // 数据为文本文件格式，具体包含字段有：
-- // (1）用户ID，字符串类型
-- // (2）时间戳，秒级别，Long类型
-- // (3）事件CODE，字符串类型，包含startUp、login、searchGoods等15个事件
-- // (4）事件名称，字符串类型，包含启动、登陆、搜索商品等15个事件
-- // (5）事件属性，Json串格式 。包含，city：字符串；name:字符串；brand:字符串；price:浮点型（3位精度），nums：整型,how：整型；
-- // (6）日期，字符串类型
-- // 测试数据总条数3亿左右，日期范围：2018/06/01到2018/07/05。
-- // 比赛数据总条数10亿左右, 日期范围：2018/06/01到2018/07/15。

--  单表
DROP TABLE if exists t_event;

CREATE TABLE if not exists default.t_event on cluster logs 
(
    uid String, 
    its UInt32, 
    action_code Enum8('evaluationGoods' = 1, 'browseGoods' = 2, 'viewOrder' = 3, 'startUp' = 4, 'unsubscribeGoods' = 5, 'login' = 6, 'order' = 7, 'reminding' = 8, 'addCart' = 9, 'consultGoods' = 10, 'searchGoods' = 11, 'shareGoods' = 12, 'orderPayment' = 13, 'collectionGoods' = 14, 'confirm' = 15), 
    action_name String, 
    event_city String, 
    event_name String, 
    event_brand String, 
    event_price Float32, 
    event_nums Int8, 
    event_how Int8, 
    day Date
) 
ENGINE = MergeTree PARTITION BY toYYYYMMDD(day) ORDER BY (uid, its, action_code) SETTINGS index_granularity = 8192




-- 分布式表
DROP TABLE if exists dis_event on cluster logs;
CREATE TABLE dis_event  on cluster logs as t_event  ENGINE = Distributed(logs, default, t_event, metroHash64(uid));


-- ┌─action_code──────┬───count()─┐
-- │ evaluationGoods  │  12936434 │
-- │ browseGoods      │ 103468552 │
-- │ viewOrder        │  10344574 │
-- │ startUp          │  12933086 │
-- │ unsubscribeGoods │   1292984 │
-- │ login            │  11644274 │
-- │ order            │  19397430 │
-- │ reminding        │   2587475 │
-- │ addCart          │  23285290 │
-- │ consultGoods     │  12941405 │
-- │ searchGoods      │  51763076 │
-- │ shareGoods       │   6468159 │
-- │ orderPayment     │  18108824 │
-- │ collectionGoods  │  12934357 │
-- │ confirm          │  12930870 │
-- └──────────────────┴───────────┘