
import logging
format_str = '%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s'
format = logging.Formatter(format_str)
logging.basicConfig(level=logging.DEBUG, format=format_str)

logger = logging.getLogger()

handler = logging.FileHandler('./log/forecast.log', 'a')
handler.setFormatter(format)
logger.addHandler(handler)



import functools
import time
def timed(logger=logger, level=None, format='%s: %s ms', paras=True):
    if level is None:
        level = logging.DEBUG


    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            start = time.time()
            import pandas as pd
            args_mini = [item for item in args
                         if (type(item) in (tuple, list, dict) and len(item) <= 20)
                            or type(item) not in (tuple, list, dict, pd.DataFrame, pd.SparseDataFrame)
                         ]

            if paras:
                logger.info("Begin to run %s with:%r, %r" % (fn.__name__, args_mini, kwargs))
            else:
                logger.info(f"Begin to run {fn.__name__} with {len(args) + len(kwargs)} paras")
            result = fn(*args, **kwargs)
            duration = time.time() - start
            logging.info('cost:%7.2f sec: ===%r end (%r, %r) ' % (duration, fn.__name__, args_mini, kwargs, ))
            #logger.log(level, format, repr(fn), duration * 1000)
            return result
        return inner

    return decorator