import pandas as pd

def user_item_freq(flatfile, order_products_gesamt, orders, which_set):
    '''
    Calculate the frequency with which each user has ordered an item (absolute and relative to all orders)

    :param flatfile: pandas dataframe flatfile
    :param order_products_gesamt: dataframe containing orders and products
    :param orders: dataframe containing orders
    :param which_set: 'train' oder 'test'

    :return: flatfile with merged feature
    '''
    # only the orders before train and test are interesting
    orders = orders.loc[orders.eval_set != 'test']

    # for flatfile train to calculate frequency only prior orders can be used
    if which_set == 'train':
        order_products_gesamt = order_products_gesamt.loc[order_products_gesamt.eval_set != 'train']
        orders = orders.loc[orders.eval_set != 'train']

    orders_per_user_item = order_products_gesamt.groupby(
        ['user_id', 'product_id']).size().reset_index(name='user_item_order_freq')

    orders_per_user = orders.groupby(["user_id"]).size().reset_index(name="sum_orders_user")

    df = orders_per_user_item.merge(orders_per_user, how="left", on="user_id")
    df["user_item_order_rel"] = df["user_item_order_freq"] / df["sum_orders_user"]
    df = df.drop("sum_orders_user", 1)

    flatfile = flatfile.merge(df, how='left', on=['user_id', 'product_id'])

    return flatfile


def general_item_freq(flatfile, order_products_gesamt, which_set):
    '''
    Calculate the general frequency of ordered items and whether they
    belong to the top 1 or top 10 percent of most ordered items

    :param flatfile: pandas dataframe flatfile
    :param order_products_gesamt: dataframe containing orders and products
    :param which_set: 'train' oder 'test'

    :return: flatfile with merged features
    '''
    if which_set == 'train':
        order_products_gesamt = order_products_gesamt.loc[order_products_gesamt.eval_set != 'train']

    # calculate absolute frequency of item
    df = order_products_gesamt.groupby("product_id").size().reset_index(name="sum_prod_orders")
    df = df.sort_values("sum_prod_orders", ascending=False).reset_index(drop=True)
    df["row_number"] = df.index.values
    df["quantile"] = df["row_number"] / df.shape[0]

    def top_item(x, n=0.1):
        if (x <= n):
            return 1
        else:
            return 0

    df["top_10per_ordered_item"] = df["quantile"].apply(top_item, n=0.1)
    df["top_1per_ordered_item"] = df["quantile"].apply(top_item, n=0.01)

    df = df.drop(["row_number", "quantile"], 1)

    flatfile = flatfile.merge(df, how="left", on="product_id")

    return flatfile


def last_ord_of_item(order_product_gesamt, which_set):
    '''
    The last order of an item is calculated. For the training set the training data must be excluded. After calculating
    the last order, keys such as order_id is joined to the orders

    :param order_product_gesamt: dataframe containing orders and products
    :param which_set: 'train' or 'test'

    :return: A dataframe with the last order of a user,item and the id of that order
    '''
    if which_set == 'train':
        order_product_gesamt = order_product_gesamt.loc[order_product_gesamt.eval_set != 'train']

    last_order_item = order_product_gesamt.groupby(['user_id', 'product_id']).agg({'order_number': 'max'}).reset_index()
    # get the order ids for each order of each user
    id_of_each_order = order_product_gesamt.loc[:,
                       ['user_id', 'order_number', 'order_id']].drop_duplicates().reset_index(drop=True)
    last_order_item = last_order_item.merge(id_of_each_order, how="inner", on=['user_id', 'order_number'])
    last_order_item = last_order_item.rename(columns={'order_number': 'last_order_number', 'order_id': 'last_order_id'})
    return (last_order_item)


def time_to_last_ord(flatfile, orders, which_set):
    '''
    For every user,item get the last order before the train order and calculate the distance in days to the flatfile order

    :param flatfile: pandas dataframe flatfile
    :param orders: dataframe containing orders
    :param which_set: 'train' oder 'test'

    :return: flatfile with merged feature
    '''
    orders = orders.loc[orders.eval_set != "test"]

    orders = orders.set_index('order_number').sort_index(ascending=False)

    orders['order_days_cumsum'] = orders.groupby(
        'user_id')['days_since_prior_order'].transform(pd.Series.cumsum)
    orders['order_days_cumsum'] = orders.groupby('user_id')['order_days_cumsum'].shift(periods=1).fillna(value=0)
    orders = orders.reset_index()

    orders = orders.loc[:, ['order_id', 'order_days_cumsum']]

    flatfile = flatfile.merge(orders, how='left', left_on='last_order_id', right_on='order_id')

    return (flatfile)
