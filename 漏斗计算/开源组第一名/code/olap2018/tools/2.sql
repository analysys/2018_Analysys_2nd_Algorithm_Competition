SELECT
    day,
    city,
    countIf(level >= 1) AS _1,
    countIf(level >= 2) AS _2,
    countIf(level >= 3) AS _3,
    medianExactIf( toFloat32(x[2].1 - x[1].1), level >= 2) AS median1,
    medianExactIf( toFloat32(x[3].1 - x[2].1), level >= 3) AS median2
FROM
(
    SELECT
        x[1].2 AS day,
        length(x) AS level,
        x,
        x[1].3 as city
    FROM
    (
        SELECT arrayJoin(xFunnel(604800, 3, '')((its, day, event_city), action_code = 'searchGoods', action_code = 'consultGoods', action_code = 'order')) AS x
        FROM dis_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-06-30') AND (action_code IN ('searchGoods', 'consultGoods', 'order'))
        GROUP BY uid
    )
)
GROUP BY day, city order by city, day

