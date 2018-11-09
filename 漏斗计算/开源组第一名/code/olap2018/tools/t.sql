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
)  ENGINE = MergeTree
PARTITION BY toYYYYMMDD(day)
ORDER BY (uid, its, action_code)
SETTINGS index_granularity = 8192;

