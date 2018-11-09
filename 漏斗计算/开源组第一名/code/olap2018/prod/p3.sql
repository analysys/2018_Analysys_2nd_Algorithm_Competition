SELECT
    day,
    event_name,
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
        x[2].3 as event_name
    FROM
    (
        SELECT arrayJoin(xFunnel(604800, 3, '')((its, day, event_name), action_code = 'searchGoods', action_code = 'consultGoods', action_code = 'order')) AS x
        FROM dis_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-06-30') AND (action_code IN ('searchGoods', 'consultGoods', 'order'))
        GROUP BY uid
    )
)
GROUP BY day, event_name order by event_name, day