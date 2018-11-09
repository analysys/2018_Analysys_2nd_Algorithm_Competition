SELECT
    day,
    event_brand,
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
        x,
        x[4].3 as event_city
    FROM
    (
        SELECT arrayJoin(xFunnel(2592000, 3, '2.4=4.4')((its, day, event_city, event_brand), action_code = 'login', action_code = 'consultGoods', action_code = 'confirm', action_code = 'unsubscribeGoods')) AS x
        FROM dis_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-07-10') AND (action_code IN ('login', 'consultGoods', 'confirm', 'unsubscribeGoods'))
        GROUP BY uid
    )
)
GROUP BY day, event_city order by event_city, day 
