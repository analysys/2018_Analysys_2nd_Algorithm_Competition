-- 示例SQL
-- 1、	计算出20180601-20180610范围内，依次有序触发“login-登陆”、“searchGoods-搜索商品”、“consultGoods-咨询商品”、“addCart-加入购物车”的用户转换情况，且满足时间窗口为1天，且要求“consultGoods-咨询商品”与“addCart-加入购物车”对应的“name-商品名称”属性相同。

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
        SELECT arrayJoin(xFunnel(86400, 2, '3.3=4.3')((its, day,  event_name), action_code = 'login', action_code = 'searchGoods', action_code = 'consultGoods', action_code = 'addCart')) AS x
        FROM t_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-06-10') AND (action_code IN ('login', 'searchGoods', 'consultGoods', 'addCart'))
        GROUP BY uid
    )
)
GROUP BY day order by day 



(0,List(141545.0, 84960.0, 25689.0, 5211.0, 12226.0, 2320.0, 4429.0))
(1,List(137866.0, 91890.0, 35099.0, 9760.0, 38043.5, 5255.0, 11186.5))
(2,List(168140.0, 127585.0, 56957.0, 17991.0, 48964.0, 5076.0, 11439.0))
(3,List(247375.0, 183694.0, 82971.0, 27389.0, 42536.5, 4790.0, 11375.0))
(4,List(234834.0, 177806.0, 84035.0, 29646.0, 45086.0, 4514.0, 11054.0))
(5,List(258704.0, 196950.0, 94243.0, 33685.0, 40409.5, 4434.0, 11399.0))
(6,List(263156.0, 198872.0, 94559.0, 33683.0, 40610.0, 4406.0, 11230.0))
(7,List(220434.0, 142402.0, 56955.0, 17692.0, 18416.5, 3857.0, 9019.5))
(8,List(85520.0, 56122.0, 22332.0, 6601.0, 37397.0, 5089.5, 11603.0))
(9,List(90089.0, 47764.0, 14443.0, 3344.0, 8609.0, 2564.0, 5788.0))
 5019 4429
 
────────day─┬─────_1─┬─────_2─┬────_3─┬────_4─┬─median1─┬─median2─┬─median3─┐
│ 2018-06-01 │ 141545 │  84960 │ 25689 │  5211 │   12226 │    2320 │    4429 │
│ 2018-06-02 │ 137866 │  91890 │ 35099 │  9760 │ 38043.5 │    5255 │ 11186.5 │
│ 2018-06-03 │ 168140 │ 127585 │ 56957 │ 17991 │   48964 │    5076 │   11439 │
│ 2018-06-04 │ 247375 │ 183694 │ 82971 │ 27389 │ 42536.5 │    4790 │   11375 │
│ 2018-06-05 │ 234834 │ 177806 │ 84035 │ 29646 │   45086 │    4514 │   11054 │
│ 2018-06-06 │ 258704 │ 196950 │ 94243 │ 33685 │ 40409.5 │    4434 │   11399 │
│ 2018-06-07 │ 263156 │ 198872 │ 94559 │ 33683 │   40610 │    4406 │   11230 │
│ 2018-06-08 │ 220434 │ 142402 │ 56955 │ 17692 │ 18416.5 │    3857 │  9019.5 │
│ 2018-06-09 │  85520 │  56122 │ 22332 │  6601 │   37397 │  5089.5 │   11603 │
│ 2018-06-10 │  90089 │  47764 │ 14443 │  3344 │    8609 │    2564 │    5788 │
└────────────┴────────┴────────┴───────┴───────┴─────────┴─────────┴─────────┘

 
-- 2、	计算出2018年6月份中，依次有序触发 “searchGoods-搜索商品”、“consultGoods咨询商品”、“order-生成订单”的用户转换情况以及各步骤转换时间中位数，且满足时间窗口为7天，且结果按“searchGoods-搜索商品”的“城市”属性分组。

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
        FROM t_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-06-30') AND (action_code IN ('searchGoods', 'consultGoods', 'order'))
        GROUP BY uid
    )
)
GROUP BY day, city order by city, day


2018-06-01	上海	94901	73719	60925	221369	37726
2018-06-02	上海	88543	70510	59749	158790	32918
2018-06-03	上海	107175	80850	66066	113569	30330.5
2018-06-04	上海	147447	103327	80810	100201	29441
2018-06-05	上海	140859	99100	78040	90886	27915.5
2018-06-06	上海	149262	103751	81099	87255	28323
。。。

-- 3、	计算出2018年6月份中，依次有序触发“searchGoods-搜索商品”、“consultGoods咨询商品”、“addCart-加入购物车”、“orderPayment-订单付款”的用户转换情况以及各步骤转换时间中位数，且满足时间窗口为7天，且“consultGoods咨询商品”、“addCart-加入购物车”的“brand-品牌”相等，且结果按“consultGoods-咨询商品”的商品价格进行分组，价格分层条件为100元以下,【100-200），【200-300）,300以上。

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
        FROM t_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-06-30') AND (action_code IN ('searchGoods', 'consultGoods', 'addCart', 'orderPayment'))
        GROUP BY uid
    )
) group by day, price_level order by price_level,day


(3000以上:0,List(220576, 220576, 93774, 37668, 371242, 47613, 3762))(3000以上:1,List(212446, 212446, 96165, 39630, 324460, 46957, 4100))(3000以上:10,List(231063, 231063, 101451, 41478, 256425, 46505, 3624))(3000以上:11,List(306282, 306282, 126380, 50999, 253654, 47659, 3876))(3000以上:12,List(321350, 321350, 135483, 55048, 292405, 45295, 3791))(3000以上:13,List(327567, 327567, 138514, 56363, 392714, 45046, 3827))(3000以上:14,List(335495, 335495, 142603, 58306, 356421, 45334, 3950))


2018-06-01	3000以上	220307	220307	152012	125104	261452	84239	23547.5



-- 测试场景四：
-- 计算出20180601-20180607范围内，依次有序触发“login-登陆”、“searchGoods-搜索窗口”、“addCart-加入购物车”、”shareGoods-分享商品“的用户转换情况，且满足时间窗口为3天，且要求“searchGoods-搜索窗口”与“addCart-加入购物车”对应的“city-城市名称”属性相同

SELECT
    day, 
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
        x
    FROM
    (
        SELECT arrayJoin(xFunnel(259200, 2, '2.3=3.3')((its, day, event_city), action_code = 'login', action_code = 'searchGoods', action_code = 'addCart', action_code = 'shareGoods')) AS x
        FROM t_event
        WHERE (day >= '2018-06-01') AND (day <= '2018-06-07') AND (action_code IN ('login', 'searchGoods', 'addCart', 'shareGoods'))
        GROUP BY uid
    )
) group by day order by day


┌────────day─┬─────_1─┬─────_2─┬─────_3─┬────_4─┬──median1─┬─median2─┬─median3─┐
│ 2018-06-01 │ 141545 │ 118673 │  64269 │ 26521 │   137604 │   36351 │   13296 │
│ 2018-06-02 │ 137866 │ 124706 │  82861 │ 41876 │   137332 │   35286 │ 14092.5 │
│ 2018-06-03 │ 168140 │ 152286 │ 102847 │ 54231 │ 109087.5 │   35489 │   13943 │
│ 2018-06-04 │ 247375 │ 220846 │ 147272 │ 77996 │ 107323.5 │   35351 │ 13707.5 │
│ 2018-06-05 │ 234834 │ 207715 │ 134201 │ 67601 │    88441 │   28182 │   11111 │
│ 2018-06-06 │ 258704 │ 212895 │ 113250 │ 46305 │    60606 │   17575 │    7828 │
│ 2018-06-07 │ 263156 │ 160294 │  52889 │ 14681 │    10304 │    6995 │    3735 │
└────────────┴────────┴────────┴────────┴───────┴──────────┴─────────┴─────────┘
