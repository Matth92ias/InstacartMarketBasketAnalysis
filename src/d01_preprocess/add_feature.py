import pandas as pd

def avg_to_card_order(flatfile, order_products_gesamt, which_set):
    '''
    Calculate the average to card order of each item and add it to flatfile

    :param flatfile: pandas dataframe flatfile
    :param order_products_gesamt: dataframe containing orders and products
    :param which_set: 'train' or 'test'

    :return:  flatfile with feature merged to it
    '''
    # only the orders before train and test are interesting
    if which_set == 'train':
        order_products_gesamt = order_products_gesamt.loc[order_products_gesamt.eval_set != 'train']

    df = order_products_gesamt.groupby(['user_id', 'product_id']).agg({"add_to_cart_order": "mean"}). \
        reset_index().rename(columns={"add_to_cart_order": "avg_to_cart_order"})

    flatfile = flatfile.merge(df, how='left', on=['user_id', 'product_id'])

    return flatfile
