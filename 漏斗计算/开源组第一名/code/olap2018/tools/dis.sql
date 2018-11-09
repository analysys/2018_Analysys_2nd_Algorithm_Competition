CREATE TABLE dis_event  on cluster logs as t_event  ENGINE = Distributed(logs, default, t_event, metroHash64(uid));
