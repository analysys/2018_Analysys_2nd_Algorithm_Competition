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



