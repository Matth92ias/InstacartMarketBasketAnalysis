import pandas as pd

def order_time_of_day_converter(x):
    '''
    convert hour (0-24) into time of day

    :param x: pandas series

    :return: hour converted into class
    '''
    if (x >= 6) & (x <= 11):
        return 'morning'
    elif (x >= 12) & (x <= 14):
        return 'midday'
    elif (x >= 15) & (x <= 17):
        return 'afternoon'
    elif (x >= 18) & (x <= 22):
        return 'evening'
    elif (x >= 23) or (x <= 5):
        return 'night'


def add_time_of_day(flatfile, order_products_gesamt, orders):
    '''
    Add time of day of order to flatfile

    :param flatfile: pandas dataframe flatfile
    :param order_products_gesamt: dataframe containing orders and products
    :param orders: dataframe containing orders

    :return: flatfile with merged feature
    '''
    orders['time_of_day'] = orders['order_hour_of_day'].apply(OrderTimeofdayConverter)
    order_products_gesamt = order_products_gesamt.merge(orders.loc[:, ["order_id", "time_of_day"]],
                                                        how='left', on="order_id")

    # flatfile needs column time_of_day to join later information
    flatfile = flatfile.drop(columns=["order_id"])
    flatfile = flatfile.merge(orders.loc[:, ["user_id", "order_id", "time_of_day"]],
                              how="left", left_on=["user_id", "ff_order_id"], right_on=["user_id", "order_id"]).drop(
        "order_id", 1)

    return flatfile, order_products_gesamt, orders


def prep_time_of_day_user(flatfile, order_products_gesamt, orders, which_set):
    '''
    Calculate how often and if a user has ordered an item which is in the flatfile

    :param flatfile: pandas dataframe flatfile
    :param order_products_gesamt: dataframe containing orders and products
    :param orders: dataframe containing orders
    :param which_set: 'train' oder 'test'

    :return: flatfile with merge features
    '''
    # filter train out of product order set
    if which_set == 'train':
        order_products_gesamt = order_products_gesamt.loc[order_products_gesamt.eval_set != 'train']

    # calculate the orders of a user of a specific product at a time during the same.
    user_tod_freq = order_products_gesamt.groupby(["user_id", "product_id", "time_of_day"]).size().reset_index(
        name="sum_user_prod_ord_tod")
    user_ord_tod_freq = orders.groupby(["user_id", "time_of_day"]).size().reset_index(name="sum_user_ord_tod")
    user_tod_freq = user_tod_freq.merge(user_ord_tod_freq, how="left", on=["user_id", "time_of_day"])

    # Set this in relation to the orders a user has done during a specific time of day.
    user_tod_freq["user_prod_ord_tod_rel"] = user_tod_freq["sum_user_prod_ord_tod"] / user_tod_freq["sum_user_ord_tod"]
    user_tod_freq = user_tod_freq.drop("sum_user_ord_tod", 1)

    # join the calculated features to the flatfile
    flatfile = flatfile.merge(user_tod_freq, how="left", on=["user_id", "product_id", "time_of_day"])

    # If missing (item hasnt been ordered at a specific time of day) then fill with 0
    flatfile["sum_user_prod_ord_tod"] = flatfile["sum_user_prod_ord_tod"].fillna(0)
    flatfile["user_prod_ord_tod_rel"] = flatfile["user_prod_ord_tod_rel"].fillna(0)

    return flatfile


def prep_time_of_day_product(flatfile, order_products_gesamt, orders, which_set):
    '''
    Calculate how often a specific product is bought at a particular time of the day

    :param flatfile:
    :param order_products_gesamt:
    :param orders:
    :param which_set:

    :return: flatfile with merged features
    '''
    # filter train out of product order set
    if which_set == 'train':
        order_products_gesamt = order_products_gesamt.loc[order_products_gesamt.eval_set != 'train']

    # how often has a product been bought in total and how often at a specific time of date
    sum_prod_orders_tod = order_products_gesamt.groupby(["product_id", "time_of_day"]).size().reset_index(
        name="sum_prod_orders_tod")
    sum_prod_orders = order_products_gesamt.groupby(["product_id"]).size().reset_index(name="sum_prod_orders")

    # times a product is bought at specific time / times a product is bought in general
    sum_prod_orders_tod = sum_prod_orders_tod.merge(sum_prod_orders, how="left", on="product_id")
    sum_prod_orders_tod["prod_orders_tod_rel"] = sum_prod_orders_tod["sum_prod_orders_tod"] / sum_prod_orders_tod[
        "sum_prod_orders"]
    # drop unnecessary columns (only ratio is needed)
    sum_prod_orders_tod = sum_prod_orders_tod.drop(["sum_prod_orders_tod", "sum_prod_orders"], 1)

    # Merge to flatfile
    flatfile = flatfile.merge(sum_prod_orders_tod, how="left", on=["product_id", "time_of_day"])

    # If missing (item hasnt been ordered at a specific time of day) then fill with 0
    flatfile["prod_orders_tod_rel"] = flatfile["prod_orders_tod_rel"].fillna(0)

    return flatfile
