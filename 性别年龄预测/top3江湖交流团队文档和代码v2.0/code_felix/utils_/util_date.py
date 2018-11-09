import datetime

from code_felix.utils_.util_log import *


def convert_week_sn(week_sn, base_time='2012-01-01'):
    date: datetime = datetime.datetime.strptime(base_time, "%Y-%m-%d")
    week_sn = int(week_sn)
    date = date + datetime.timedelta(week_sn*7)
    date_str=  date.strftime("%Y-%m-%d")
    #logger.debug("get %s base on %s, %s" % (date_str, week_sn, base_time) )
    return date_str


def get_week_sn(date):
    base_time = datetime.datetime.strptime('2012-01-01', "%Y-%m-%d")

    if type(date) is str:
        date =  datetime.datetime.strptime(date, "%Y-%m-%d")

    gap = (date - base_time ).days

    return gap//7


def convert_date(time, input_type=None):
    import datetime
    #"%Y-%m-%d %H:%M:%S'
    format = "%Y-%m-%d %H:%M:%S"
    try:
        if time is None:
            return None
        elif type(time) is str or input_type == 'str':
            format = format[0:len(time)-2]
            return datetime.datetime.strptime(time,format )
        else:
            format = format[0:8]
            return time.strftime(format)
    except :
        logger.warning("Error convert time:%s with%s" % (time,format) )
        return None



def convert_monday(date):
    if date is None:
        return None

    input_type = 'date'
    if type(date) is str:
        date = convert_date(date)
        input_type = 'str'


    day_sn = date.weekday()
    if day_sn != 0: #not monday
        date = date - datetime.timedelta(day_sn )

    #logger.debug('Convert date sunday:%s' % date)
    if input_type == 'date' :
        return date
    else :
        return convert_date(date)

def convert_sunday(date):
    if date is None:
        return None

    input_type = 'date'
    if type(date) is str:
        date = convert_date(date)
        input_type = 'str'


    day_sn = date.weekday()
    if day_sn != 6: #not sunday
        date = date + datetime.timedelta( 6 - day_sn )

    #logger.debug('Convert date sunday:%s' % date)
    if input_type == 'date' :
        return date
    else :
        return convert_date(date)


def datetime_offset_by_month(datetime1, n = 1):
    from monthdelta import monthdelta

    if type(datetime1) is str:
        datetime1 = convert_date(datetime1)
    return datetime1 + monthdelta(n)

def convert_to_month_end(date):
    if type(date) is str:
        date = convert_date(date)

    dYear = date.strftime("%Y")  # get the year
    dMonth = str(int(date.strftime("%m")) % 12 + 1)  # get next month, watch rollover
    dDay = "1"  # first day of next month
    nextMonth = convert_date("%s-%02d-%02d" % (dYear, int(dMonth), int(dDay)))  # make a datetime obj for 1st of next month
    delta = datetime.timedelta(days=1)  # create a delta of 1 second
    return nextMonth - delta

def shift_month_4_csm(range_to,shift):
    from monthdelta import monthdelta
    original_type = type(range_to)

    range_to = convert_date(range_to, datetime)
    range_to = range_to - monthdelta(shift)
    range_to = datetime.datetime(range_to.year, range_to.month, 1)
    range_to = convert_sunday(range_to)

    if original_type is str:
        return convert_date(range_to)
    else:
        return range_to



if __name__ == '__main__':
    print(shift_month_4_csm("2017-07-01", 2))
    print(shift_month_4_csm("2017-07-01", 3))
    print(shift_month_4_csm("2017-07-01", 4))
    print(shift_month_4_csm("2017-07-01", 5))
    print(shift_month_4_csm("2017-07-01", 6))
    pass
    #update_email_notification_status()
    #logging.debug(get_suspect_list(5))
    #update_email_notification_status()
    #convert_week_sn(0)
    #convert_week_sn(1)
    #convert_week_sn(2)
    #print(get_email_list(5))
    #print(get_site_predict_history('458430'))
    #print(get_week_sn('2012-01-08'))
    #get_suspect_list(2)
    #print(get_contract_info(414866,'2016-12-11','2017-04-30'))

    #print(convert_to_month_end('2012-01-02'))
    #print(convert_date('2017-05-13 00:00:00'))
    #print(convert_date('2012-01'))

    #print(verify_is_group('zhiysong','###','###'))

