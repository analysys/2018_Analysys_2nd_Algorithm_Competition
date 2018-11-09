SELECT
    day,
    multiIf(level < 2, '没有价格',  price < 1000, '1000元以下',  price < 2000 AND price >= 1000, '1000-2000',  price < 3000 AND price >= 2000, '2000-3000', '3000以上' ) as price_level,
    countIf(level >= 1) AS _1,
    countIf(level >= 2) AS _2,
    countIf(level >= 3) AS _3,
    countIf(level >= 4) AS _4,
    medianExactIf(  toFloat32(x[2].1 - x[1].1), level >= 2 ) as median1,
    medianExactIf(  toFloat32(x[3].1 - x[2].1), level >= 3 ) as median2,
    medianExactIf(  toFloat32(x[4].1 - x[3].1), level >= 4 ) as median3
FROM
(
    SELECT
        x[1].2 AS day,
        length(x) AS level,
        x,
        x[2].3 as price
    FROM
    (
        SELECT arrayJoin(xFunnel(604800, 3, '2.4=3.4')((its, day, event_price, event_brand), action_code = 'searchGoods', action_code = 'consultGoods', action_code = 'addCart', action_code = 'orderPayment')) AS x
        FROM dis_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-06-30') AND (action_code IN ('searchGoods', 'consultGoods', 'addCart', 'orderPayment'))
        GROUP BY uid
    )
) group by day, price_level order by price_level,day

